//! Post-quantum base OT using ML-KEM.

use std::io;

use cryprot_core::{Block, buf::Buf, rand_compat::RngCompat, random_oracle::RandomOracle};
use cryprot_net::{Connection, ConnectionError};
use futures::{SinkExt, StreamExt};
// ML-KEM variant: change to MlKem512/MlKem512Params or MlKem768/MlKem768Params
// for different security levels.
use ml_kem::{
    Ciphertext as MlKemCiphertext, EncodedSizeUser, KemCore, MlKem1024 as MlKem,
    MlKem1024Params as MlKemParams, SharedKey,
    array::typenum::Unsigned,
    kem::{Decapsulate, DecapsulationKey, Encapsulate, EncapsulationKey as MlKemEncapsulationKey},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use subtle::{Choice, ConditionallySelectable};
use tracing::Level;

use crate::{Connected, RotReceiver, RotSender, SemiHonest, phase};

const ENCAPSULATION_KEY_LEN: usize =
    <MlKemEncapsulationKey<MlKemParams> as EncodedSizeUser>::EncodedSize::USIZE;
const CIPHERTEXT_LEN: usize = <MlKem as KemCore>::CiphertextSize::USIZE;
const HASH_DOMAIN_SEPARATOR: &[u8] = b"MlKemOt";

// ML-KEM polynomial arithmetic constants for the MR19 protocol.
// The encapsulation key is encoded as ek = ByteEncode₁₂(t̂) ‖ ρ,
// where t̂ is a vector of k polynomials in NTT domain (k=4 for ML-KEM-1024)
// and ρ is a 32-byte seed for the public matrix A.
const Q: u16 = 3329;
const RHO_BYTES: usize = 32;
const T_HAT_BYTES: usize = ENCAPSULATION_KEY_LEN - RHO_BYTES;
const NUM_COEFFS: usize = T_HAT_BYTES * 2 / 3;
const MR19_HASH_DOMAIN: &[u8] = b"MlKemOtMR19";

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("quic connection error")]
    Connection(#[from] ConnectionError),
    #[error("io communication error")]
    Io(#[from] io::Error),
    #[error(
        "invalid count of keys/ciphertexts received. expected: {expected}, actual0: {actual0}, actual1: {actual1}"
    )]
    InvalidDataCount {
        expected: usize,
        actual0: usize,
        actual1: usize,
    },
    #[error("expected message but stream is closed")]
    ClosedStream,
    #[error("ML-KEM decapsulation failed")]
    Decapsulation,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct EncapKeyBytes(#[serde(with = "serde_bytes")] [u8; ENCAPSULATION_KEY_LEN]);

impl ConditionallySelectable for EncapKeyBytes {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(<[u8; ENCAPSULATION_KEY_LEN]>::conditional_select(
            &a.0, &b.0, choice,
        ))
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct CtBytes(#[serde(with = "serde_bytes")] [u8; CIPHERTEXT_LEN]);

impl ConditionallySelectable for CtBytes {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(<[u8; CIPHERTEXT_LEN]>::conditional_select(
            &a.0, &b.0, choice,
        ))
    }
}

// Message from receiver to sender: two values (r_0, r_1) per OT.
// MR19 protocol: sender reconstructs pk_i = r_i + H(r_{1-i}).
#[derive(Serialize, Deserialize)]
struct EncapsulationKeysMessage {
    eks0: Vec<EncapKeyBytes>,
    eks1: Vec<EncapKeyBytes>,
}

// Message from sender to receiver: two ciphertexts per OT.
#[derive(Serialize, Deserialize)]
struct CiphertextsMessage {
    cts0: Vec<CtBytes>,
    cts1: Vec<CtBytes>,
}

pub struct MlKemOt {
    rng: StdRng,
    conn: Connection,
}

/// Note: MlKemOt is not `Malicious` secure in itself.
impl SemiHonest for MlKemOt {}

impl MlKemOt {
    pub fn new(connection: Connection) -> Self {
        Self::new_with_rng(connection, StdRng::from_os_rng())
    }

    pub fn new_with_rng(connection: Connection, rng: StdRng) -> MlKemOt {
        Self {
            conn: connection,
            rng,
        }
    }
}

impl Connected for MlKemOt {
    fn connection(&mut self) -> &mut Connection {
        &mut self.conn
    }
}

impl RotSender for MlKemOt {
    type Error = Error;

    #[tracing::instrument(level = Level::DEBUG, skip_all, fields(count = ots.len()))]
    #[tracing::instrument(target = "cryprot_metrics", level = Level::TRACE, skip_all, fields(phase = phase::BASE_OT))]
    async fn send_into(&mut self, ots: &mut impl Buf<[Block; 2]>) -> Result<(), Self::Error> {
        let count = ots.len();
        let (mut send, mut recv) = self.conn.byte_stream().await?;

        let receiver_msg: EncapsulationKeysMessage = {
            let mut recv_stream = recv.as_stream();
            recv_stream.next().await.ok_or(Error::ClosedStream)??
        };

        if receiver_msg.eks0.len() != count || receiver_msg.eks1.len() != count {
            return Err(Error::InvalidDataCount {
                expected: count,
                actual0: receiver_msg.eks0.len(),
                actual1: receiver_msg.eks1.len(),
            });
        }

        let mut cts0 = Vec::with_capacity(count);
        let mut cts1 = Vec::with_capacity(count);
        for (i, (r0, r1)) in receiver_msg
            .eks0
            .iter()
            .zip(receiver_msg.eks1.iter())
            .enumerate()
        {
            // MR19: reconstruct encapsulation keys from (r_0, r_1).
            // pk_0 = r_0 + H(r_1),  pk_1 = r_1 + H(r_0)
            let pk0 = EncapKeyBytes(reconstruct_ek(&r0.0, &r1.0));
            let pk1 = EncapKeyBytes(reconstruct_ek(&r1.0, &r0.0));

            let (ct0, key0) = encapsulate(&pk0, &mut self.rng);
            let key0 = hash(&key0, i);

            let (ct1, key1) = encapsulate(&pk1, &mut self.rng);
            let key1 = hash(&key1, i);

            cts0.push(ct0);
            cts1.push(ct1);
            ots[i] = [key0, key1];
        }

        let sender_msg = CiphertextsMessage { cts0, cts1 };
        {
            let mut send_stream = send.as_stream();
            send_stream.send(sender_msg).await?;
        }

        Ok(())
    }
}

impl RotReceiver for MlKemOt {
    type Error = Error;

    #[tracing::instrument(level = Level::DEBUG, skip_all, fields(count = ots.len()))]
    #[tracing::instrument(target = "cryprot_metrics", level = Level::TRACE, skip_all, fields(phase = phase::BASE_OT))]
    async fn receive_into(
        &mut self,
        ots: &mut impl Buf<Block>,
        choices: &[Choice],
    ) -> Result<(), Self::Error> {
        let count = ots.len();
        assert_eq!(choices.len(), count);

        let (mut send, mut recv) = self.conn.byte_stream().await?;

        let mut decap_keys: Vec<DecapsulationKey<MlKemParams>> = Vec::with_capacity(count);
        let mut eks0 = Vec::with_capacity(count);
        let mut eks1 = Vec::with_capacity(count);

        for choice in choices.iter() {
            // Generate real keypair.
            let (dk, ek) = MlKem::generate(&mut RngCompat(&mut self.rng));
            let ek_bytes: [u8; ENCAPSULATION_KEY_LEN] = ek
                .as_bytes()
                .as_slice()
                .try_into()
                .expect("incorrect encapsulation key size");

            // MR19 protocol: construct (r_b, r_{1-b}) such that
            // r_b + H(r_{1-b}) = ek (on the t̂ polynomial vector, with shared ρ).
            let rho = &ek_bytes[T_HAT_BYTES..];

            // Generate a random fake ek sharing the same ρ (same matrix A).
            let fake_t_hat = random_coeffs(&mut self.rng);
            let fake_ek = assemble_ek(&fake_t_hat, rho);

            // r_b = ek - H(fake_ek) on t̂ coefficients.
            let t_hat_real = decode_t_hat(&ek_bytes[..T_HAT_BYTES]);
            let h_fake = hash_ek_to_coeffs(&fake_ek);
            let r_b_t_hat = sub_mod_q(&t_hat_real, &h_fake);
            let r_b_bytes = assemble_ek(&r_b_t_hat, rho);

            let r_b = EncapKeyBytes(r_b_bytes);
            let r_1_minus_b = EncapKeyBytes(fake_ek);

            // Constant-time selection based on choice bit.
            // choice=0: ek0=r_b (r_0), ek1=r_{1-b} (r_1) → pk_0 = r_0+H(r_1) = ek
            // choice=1: ek0=r_{1-b} (r_0), ek1=r_b (r_1) → pk_1 = r_1+H(r_0) = ek
            let ek0 = EncapKeyBytes::conditional_select(&r_b, &r_1_minus_b, *choice);
            let ek1 = EncapKeyBytes::conditional_select(&r_1_minus_b, &r_b, *choice);

            decap_keys.push(dk);
            eks0.push(ek0);
            eks1.push(ek1);
        }

        let receiver_msg = EncapsulationKeysMessage { eks0, eks1 };
        {
            let mut send_stream = send.as_stream();
            send_stream.send(receiver_msg).await?;
        }

        let sender_msg: CiphertextsMessage = {
            let mut recv_stream = recv.as_stream();
            recv_stream.next().await.ok_or(Error::ClosedStream)??
        };

        if sender_msg.cts0.len() != count || sender_msg.cts1.len() != count {
            return Err(Error::InvalidDataCount {
                expected: count,
                actual0: sender_msg.cts0.len(),
                actual1: sender_msg.cts1.len(),
            });
        }

        // Decapsulate the chosen ciphertext for each OT.
        for (i, ((dk, choice), (ct0, ct1))) in decap_keys
            .iter()
            .zip(choices.iter())
            .zip(sender_msg.cts0.iter().zip(sender_msg.cts1.iter()))
            .enumerate()
        {
            let chosen_ct: MlKemCiphertext<MlKem> =
                CtBytes::conditional_select(ct0, ct1, *choice).0.into();
            let shared_key = dk
                .decapsulate(&chosen_ct)
                .map_err(|_| Error::Decapsulation)?;
            let shared_key = hash(&shared_key, i);
            ots[i] = shared_key;
        }

        Ok(())
    }
}

// === MR19 polynomial arithmetic helpers ===
// These operate on the t̂ portion of ML-KEM encapsulation keys, which is
// encoded using ByteEncode₁₂ (FIPS 203): each 3 bytes encode 2 coefficients
// of 12 bits each, with all coefficients in [0, q).

/// Decode ByteEncode₁₂: 3 bytes → 2 coefficients (12 bits each), reduced mod q.
fn decode_t_hat(bytes: &[u8]) -> [u16; NUM_COEFFS] {
    debug_assert_eq!(bytes.len(), T_HAT_BYTES);
    let mut coeffs = [0u16; NUM_COEFFS];
    for (i, chunk) in bytes.chunks_exact(3).enumerate() {
        let d0 = chunk[0] as u16;
        let d1 = chunk[1] as u16;
        let d2 = chunk[2] as u16;
        coeffs[2 * i] = (d0 | ((d1 & 0x0F) << 8)) % Q;
        coeffs[2 * i + 1] = ((d1 >> 4) | (d2 << 4)) % Q;
    }
    coeffs
}

/// Encode coefficients as ByteEncode₁₂: 2 coefficients → 3 bytes.
fn encode_t_hat(coeffs: &[u16; NUM_COEFFS]) -> [u8; T_HAT_BYTES] {
    let mut bytes = [0u8; T_HAT_BYTES];
    for (i, pair) in coeffs.chunks_exact(2).enumerate() {
        let a = pair[0];
        let b = pair[1];
        bytes[3 * i] = (a & 0xFF) as u8;
        bytes[3 * i + 1] = ((a >> 8) | ((b & 0x0F) << 4)) as u8;
        bytes[3 * i + 2] = (b >> 4) as u8;
    }
    bytes
}

/// Add two coefficient vectors element-wise mod q.
fn add_mod_q(
    a: &[u16; NUM_COEFFS],
    b: &[u16; NUM_COEFFS],
) -> [u16; NUM_COEFFS] {
    let mut result = [0u16; NUM_COEFFS];
    for i in 0..NUM_COEFFS {
        result[i] = (a[i] + b[i]) % Q;
    }
    result
}

/// Subtract two coefficient vectors element-wise mod q.
fn sub_mod_q(
    a: &[u16; NUM_COEFFS],
    b: &[u16; NUM_COEFFS],
) -> [u16; NUM_COEFFS] {
    let mut result = [0u16; NUM_COEFFS];
    for i in 0..NUM_COEFFS {
        result[i] = (a[i] + Q - b[i]) % Q;
    }
    result
}

/// Hash an encoded encapsulation key to a t̂ coefficient vector mod q.
/// Uses BLAKE3 XOF with rejection sampling (12 bits, accept if < q).
fn hash_ek_to_coeffs(ek: &[u8; ENCAPSULATION_KEY_LEN]) -> [u16; NUM_COEFFS] {
    let mut ro = RandomOracle::new();
    ro.update(MR19_HASH_DOMAIN);
    ro.update(ek);
    let mut xof = ro.finalize_xof();
    let mut coeffs = [0u16; NUM_COEFFS];
    for c in coeffs.iter_mut() {
        loop {
            let mut buf = [0u8; 2];
            xof.fill(&mut buf);
            let val = u16::from_le_bytes(buf) & 0x0FFF;
            if val < Q {
                *c = val;
                break;
            }
        }
    }
    coeffs
}

/// Generate random coefficients mod q using rejection sampling.
fn random_coeffs(rng: &mut impl Rng) -> [u16; NUM_COEFFS] {
    let mut coeffs = [0u16; NUM_COEFFS];
    for c in coeffs.iter_mut() {
        loop {
            let val: u16 = rng.random::<u16>() & 0x0FFF;
            if val < Q {
                *c = val;
                break;
            }
        }
    }
    coeffs
}

/// Assemble a full encapsulation key encoding from t̂ coefficients and ρ.
fn assemble_ek(t_hat: &[u16; NUM_COEFFS], rho: &[u8]) -> [u8; ENCAPSULATION_KEY_LEN] {
    let mut ek = [0u8; ENCAPSULATION_KEY_LEN];
    ek[..T_HAT_BYTES].copy_from_slice(&encode_t_hat(t_hat));
    ek[T_HAT_BYTES..].copy_from_slice(rho);
    ek
}

/// MR19 key reconstruction: pk = r + H(other), using ρ from r.
fn reconstruct_ek(
    r: &[u8; ENCAPSULATION_KEY_LEN],
    other: &[u8; ENCAPSULATION_KEY_LEN],
) -> [u8; ENCAPSULATION_KEY_LEN] {
    let r_coeffs = decode_t_hat(&r[..T_HAT_BYTES]);
    let h_other = hash_ek_to_coeffs(other);
    let pk_coeffs = add_mod_q(&r_coeffs, &h_other);
    assemble_ek(&pk_coeffs, &r[T_HAT_BYTES..])
}

// Encapsulates to the given key, returning the ciphertext and the shared key.
fn encapsulate(ek: &EncapKeyBytes, rng: &mut StdRng) -> (CtBytes, SharedKey<MlKem>) {
    let parsed_ek = MlKemEncapsulationKey::<MlKemParams>::from_bytes((&ek.0).into());
    let (ct, k): (MlKemCiphertext<MlKem>, SharedKey<MlKem>) = parsed_ek
        .encapsulate(&mut RngCompat(rng))
        .expect("encapsulation should not fail");
    (
        CtBytes(ct.as_slice().try_into().expect("incorrect ciphertext size")),
        k,
    )
}

// Derive an OT key from the ML-KEM shared key using a random oracle XOF,
// extracting a Block-sized (128-bit) output.
fn hash(key: &SharedKey<MlKem>, tweak: usize) -> Block {
    let mut ro = RandomOracle::new();
    ro.update(HASH_DOMAIN_SEPARATOR);
    ro.update(key.as_slice());
    ro.update(&tweak.to_le_bytes());
    let mut out = ro.finalize_xof();
    let mut block = Block::ZERO;
    out.fill(block.as_mut_bytes());
    block
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cryprot_net::testing::{init_tracing, local_conn};
    use rand::{SeedableRng, rngs::StdRng};

    use super::MlKemOt;
    use crate::{RotReceiver, RotSender, random_choices};

    #[test]
    fn encode_decode_roundtrip() {
        use super::*;
        let mut rng = StdRng::seed_from_u64(42);
        let (_, ek) = MlKem::generate(&mut RngCompat(&mut rng));
        let ek_bytes: [u8; ENCAPSULATION_KEY_LEN] =
            ek.as_bytes().as_slice().try_into().unwrap();

        // Check all coefficients are < Q
        let coeffs = decode_t_hat(&ek_bytes[..T_HAT_BYTES]);
        for (i, &c) in coeffs.iter().enumerate() {
            assert!(c < Q, "coefficient {i} = {c} >= Q={Q}");
        }

        // Check encode roundtrip
        let re_encoded = encode_t_hat(&coeffs);
        assert_eq!(
            &ek_bytes[..T_HAT_BYTES],
            &re_encoded[..],
            "encode/decode roundtrip failed"
        );
    }

    #[test]
    fn mr19_reconstruction() {
        use super::*;
        let mut rng = StdRng::seed_from_u64(42);
        let (_, ek) = MlKem::generate(&mut RngCompat(&mut rng));
        let ek_bytes: [u8; ENCAPSULATION_KEY_LEN] =
            ek.as_bytes().as_slice().try_into().unwrap();
        let rho = &ek_bytes[T_HAT_BYTES..];

        // Generate fake key
        let fake_t_hat = random_coeffs(&mut rng);
        let fake_ek = assemble_ek(&fake_t_hat, rho);

        // Compute r_b = ek - H(fake_ek)
        let t_hat_real = decode_t_hat(&ek_bytes[..T_HAT_BYTES]);
        let h_fake = hash_ek_to_coeffs(&fake_ek);
        let r_b_t_hat = sub_mod_q(&t_hat_real, &h_fake);
        let r_b_bytes = assemble_ek(&r_b_t_hat, rho);

        // Reconstruct: pk = r_b + H(fake_ek)
        let reconstructed = reconstruct_ek(&r_b_bytes, &fake_ek);
        assert_eq!(
            ek_bytes, reconstructed,
            "MR19 reconstruction failed: pk != ek"
        );
    }

    #[tokio::test]
    async fn mlkem_base_rot_random_choices() -> Result<()> {
        let _g = init_tracing();
        let (con1, con2) = local_conn().await?;
        let mut rng1 = StdRng::seed_from_u64(42);
        let rng2 = StdRng::seed_from_u64(42 * 42);
        let count = 128;
        let choices = random_choices(count, &mut rng1);

        let mut sender = MlKemOt::new_with_rng(con1, rng1);
        let mut receiver = MlKemOt::new_with_rng(con2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }

    #[tokio::test]
    async fn mlkem_base_rot_zero_choices() -> Result<()> {
        let _g = init_tracing();
        let (con1, con2) = local_conn().await?;
        let rng1 = StdRng::seed_from_u64(123);
        let rng2 = StdRng::seed_from_u64(456);
        let count = 128;
        let choices: Vec<_> = (0..count).map(|_| subtle::Choice::from(0)).collect();

        let mut sender = MlKemOt::new_with_rng(con1, rng1);
        let mut receiver = MlKemOt::new_with_rng(con2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }

    #[tokio::test]
    async fn mlkem_base_rot_one_choices() -> Result<()> {
        let _g = init_tracing();
        let (con1, con2) = local_conn().await?;
        let rng1 = StdRng::seed_from_u64(789);
        let rng2 = StdRng::seed_from_u64(101112);
        let count = 128;
        let choices: Vec<_> = (0..count).map(|_| subtle::Choice::from(1)).collect();

        let mut sender = MlKemOt::new_with_rng(con1, rng1);
        let mut receiver = MlKemOt::new_with_rng(con2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }
}
