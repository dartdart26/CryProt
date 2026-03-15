//! Post-quantum base OT using ML-KEM.
//!
//! Implements the MR19 protocol (Masny-Rindal, ePrint 2019/706, Figure 8)
//! instantiated with ML-KEM as per Section D.3.
//! See `docs/mlkem-ot-protocol.md` for the full protocol description.

use std::{io, mem::size_of};

use cryprot_core::{Block, buf::Buf, rand_compat::RngCompat, random_oracle::RandomOracle};
use cryprot_net::{Connection, ConnectionError};
use futures::{SinkExt, StreamExt};
use hybrid_array::typenum::Unsigned;
// ML-KEM variant: change to MlKem512/MlKem512Params or MlKem768/MlKem768Params
// for different security levels.
use ml_kem::{
    Ciphertext as MlKemCiphertext, EncodedSizeUser, KemCore, MlKem1024 as MlKem,
    MlKem1024Params as MlKemParams, ParameterSet, SharedKey,
    kem::{Decapsulate, DecapsulationKey, Encapsulate, EncapsulationKey as MlKemEncapsulationKey},
};
use module_lattice::{Encode, Field, NttPolynomial};
use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};
use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update, XofReader},
};
use subtle::{Choice, ConditionallySelectable};
use tracing::Level;

use crate::{Connected, RotReceiver, RotSender, SemiHonest, phase};

// Define the ML-KEM base field (q = 3329).
module_lattice::define_field!(MlKemField, u16, u32, u64, 3329);

// Module dimension derived from the chosen ML-KEM parameter set.
type K = <MlKemParams as ParameterSet>::K;

type NttVector = module_lattice::NttVector<MlKemField, K>;

type U12 = hybrid_array::typenum::U12;

const ENCAPSULATION_KEY_LEN: usize =
    <MlKemEncapsulationKey<MlKemParams> as EncodedSizeUser>::EncodedSize::USIZE;
const CIPHERTEXT_LEN: usize = <MlKem as KemCore>::CiphertextSize::USIZE;
const HASH_DOMAIN_SEPARATOR: &[u8] = b"MlKemOt";

// Number of coefficients per polynomial (FIPS 203, Section 2: n = 256).
const NUM_COEFFICIENTS: usize = 256;

/// rho is a 32-byte seed used to derive the public matrix A_hat (FIPS 203).
type Rho = [u8; 32];

// Serialized t_hat is the encapsulation key minus the rho suffix.
const T_HAT_BYTES_LEN: usize = ENCAPSULATION_KEY_LEN - size_of::<Rho>();

// ---------------------------------------------------------------------------
// Protocol helper functions (see docs/mlkem-ot-protocol.md)
// ---------------------------------------------------------------------------

/// Parse serialized encapsulation key bytes into (NttVector, rho).
/// The input is a fixed-size array so the slicing is infallible.
fn parse_ek(bytes: &[u8; ENCAPSULATION_KEY_LEN]) -> (NttVector, Rho) {
    let enc = bytes[..T_HAT_BYTES_LEN]
        .try_into()
        .expect("t_hat length mismatch");
    let t_hat = <NttVector as Encode<U12>>::decode(enc);
    let rho = bytes[T_HAT_BYTES_LEN..]
        .try_into()
        .expect("rho length mismatch");
    (t_hat, rho)
}

/// Serialize NttVector + rho back into encapsulation key bytes.
fn serialize_ek(t_hat: &NttVector, rho: &Rho) -> [u8; ENCAPSULATION_KEY_LEN] {
    let encoded = <NttVector as Encode<U12>>::encode(t_hat);
    let mut out = [0u8; ENCAPSULATION_KEY_LEN];
    out[..T_HAT_BYTES_LEN].copy_from_slice(encoded.as_slice());
    out[T_HAT_BYTES_LEN..].copy_from_slice(rho);
    out
}

/// XOF(rho, j, i) from FIPS 203, Algorithm 2 SHAKE128example.
/// In Algorithm 13 (K-PKE.KeyGen), this is called as XOF(rho, j, i) where
/// j is the column index (byte 32) and i is the row index (byte 33),
/// using 0-based indexing.
fn xof(seed: &Rho, j: u8, i: u8) -> impl XofReader {
    let mut h = Shake128::default();
    h.update(seed);
    h.update(&[i, j]);
    h.finalize_xof()
}

/// FIPS 203 Algorithm 7: SampleNTT.
/// Rejection sampling from a byte stream to produce a pseudorandom NTT
/// polynomial.
///
/// Adapted from the ml-kem crate's `sample_ntt`.
fn sample_ntt_poly(xof: &mut impl XofReader) -> NttPolynomial<MlKemField> {
    const Q: u16 = MlKemField::Q;
    // Read 32 triples (3 bytes each) at a time from the XOF.
    const BUF_LEN: usize = 96;
    let mut poly = NttPolynomial::<MlKemField>::default();
    let mut buf = [0u8; BUF_LEN];
    let mut pos = BUF_LEN; // start at end to trigger first read
    let mut i = 0;

    while i < NUM_COEFFICIENTS {
        if pos >= BUF_LEN {
            xof.read(&mut buf);
            pos = 0;
        }

        let d1 = u16::from(buf[pos]) | ((u16::from(buf[pos + 1]) & 0x0F) << 8);
        let d2 = (u16::from(buf[pos + 1]) >> 4) | (u16::from(buf[pos + 2]) << 4);
        pos += 3;

        if d1 < Q {
            poly.0[i] = module_lattice::Elem::new(d1);
            i += 1;
        }
        if i < NUM_COEFFICIENTS && d2 < Q {
            poly.0[i] = module_lattice::Elem::new(d2);
            i += 1;
        }
    }

    poly
}

/// SampleNTTVector: call SampleNTT k times with FIPS 203 domain separation.
/// Produces a pseudorandom NttVector<k> from a 32-byte seed.
/// Each polynomial j uses XOF(seed || j || 0).
fn sample_ntt_vector(seed: &Rho) -> NttVector {
    NttVector::new(
        (0..K::USIZE)
            .map(|j| {
                let mut reader = xof(seed, j as u8, 0);
                sample_ntt_poly(&mut reader)
            })
            .collect(),
    )
}

/// H(ek): hash-to-key. Maps an NttVector to another NttVector via SHA3-256.
/// Corresponds to libOTe's `pkHash`.
fn hash_to_key(t_hat: &NttVector) -> NttVector {
    use sha3::Digest;
    let encoded = <NttVector as Encode<U12>>::encode(t_hat);
    let seed: Rho = sha3::Sha3_256::digest(encoded.as_slice()).into();
    sample_ntt_vector(&seed)
}

/// RandomEK: generate a random NttVector from a random seed.
fn random_ek(rng: &mut StdRng) -> NttVector {
    let seed: Rho = rng.random();
    sample_ntt_vector(&seed)
}

// ---------------------------------------------------------------------------
// Wire types and protocol implementation
// ---------------------------------------------------------------------------

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
        Self::new_with_rng(connection, rand::make_rng())
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
        for (i, (r0_bytes, r1_bytes)) in receiver_msg
            .eks0
            .iter()
            .zip(receiver_msg.eks1.iter())
            .enumerate()
        {
            // Reconstruct encapsulation keys: ek_j = r_j + H(r_{1-j})
            let (r0, rho) = parse_ek(&r0_bytes.0);
            let (r1, _) = parse_ek(&r1_bytes.0);

            let ek0_bytes = serialize_ek(&(&r0 + &hash_to_key(&r1)), &rho);
            let ek1_bytes = serialize_ek(&(&r1 + &hash_to_key(&r0)), &rho);

            let (ct0, key0) = encapsulate(&EncapKeyBytes(ek0_bytes), &mut self.rng);
            let key0 = hash(&key0, i);

            let (ct1, key1) = encapsulate(&EncapKeyBytes(ek1_bytes), &mut self.rng);
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
            // Step 1: Generate real keypair.
            let (dk, ek) = MlKem::generate(&mut RngCompat(&mut self.rng));
            let ek_bytes: [u8; ENCAPSULATION_KEY_LEN] = ek
                .as_bytes()
                .as_slice()
                .try_into()
                .expect("incorrect encapsulation key size");
            let (real_t_hat, rho) = parse_ek(&ek_bytes);

            // Step 2: Sample random key for position 1-b.
            let rand_t_hat = random_ek(&mut self.rng);

            // Step 3: Compute correlated key for position b: r_b = ek - H(r_{1-b}).
            let correlated_t_hat = &real_t_hat - &hash_to_key(&rand_t_hat);

            // Serialize both keys with the same rho.
            let correlated_bytes = EncapKeyBytes(serialize_ek(&correlated_t_hat, &rho));
            let random_bytes = EncapKeyBytes(serialize_ek(&rand_t_hat, &rho));

            // Step 4: Select (r_0, r_1) based on choice bit (constant-time).
            // If b=0: r_0 = correlated (real side), r_1 = random.
            // If b=1: r_0 = random, r_1 = correlated (real side).
            let ek0 = EncapKeyBytes::conditional_select(&correlated_bytes, &random_bytes, *choice);
            let ek1 = EncapKeyBytes::conditional_select(&random_bytes, &correlated_bytes, *choice);

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

        // Step 10-11: Decapsulate the chosen ciphertext and derive OT key.
        for (i, ((dk, choice), (ct0, ct1))) in decap_keys
            .iter()
            .zip(choices.iter())
            .zip(sender_msg.cts0.iter().zip(sender_msg.cts1.iter()))
            .enumerate()
        {
            let ct_bytes = CtBytes::conditional_select(ct0, ct1, *choice).0;
            let chosen_ct: MlKemCiphertext<MlKem> = ct_bytes
                .as_slice()
                .try_into()
                .expect("incorrect ciphertext size");
            let shared_key = dk
                .decapsulate(&chosen_ct)
                .map_err(|_| Error::Decapsulation)?;
            let shared_key = hash(&shared_key, i);
            ots[i] = shared_key;
        }

        Ok(())
    }
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
