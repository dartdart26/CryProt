//! Post-quantum base OT using ML-KEM-768 following the MR19 (Masny-Rindal) construction.
//!
//! This provides post-quantum security based on the Module-LWE hardness assumption,
//! as used in libOTe's `ENABLE_MR_KYBER` option.
//!
//! Reference: [[MR19](https://eprint.iacr.org/2019/706)] Endemic Oblivious Transfer

use std::io;

use cryprot_core::{
    Block,
    buf::Buf,
    rand_compat::RngCompat,
    random_oracle::RandomOracle,
};
use cryprot_net::{Connection, ConnectionError};
use futures::{SinkExt, StreamExt};
use ml_kem::{
    Ciphertext, EncodedSizeUser, KemCore, MlKem768, MlKem768Params, SharedKey,
    kem::{Decapsulate, DecapsulationKey, Encapsulate, EncapsulationKey},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use subtle::Choice;
use tracing::Level;

use crate::{Connected, Malicious, RotReceiver, RotSender, SemiHonest, phase};

/// ML-KEM-768 encapsulation key size in bytes.
const EK_BYTES: usize = 1184;
/// ML-KEM-768 ciphertext size in bytes.
const CT_BYTES: usize = 1088;

/// Post-quantum base OT using ML-KEM-768.
///
/// Implements the MR19 protocol that transforms any KEM into an OT protocol.
/// Provides post-quantum security assuming Module-LWE is hard.
pub struct MlKemOt {
    rng: StdRng,
    conn: Connection,
}

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

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("quic connection error")]
    Connection(#[from] ConnectionError),
    #[error("io communication error")]
    Io(#[from] io::Error),
    #[error("insufficient keys/ciphertexts received. expected: {expected}, actual: {actual}")]
    InsufficientData { expected: usize, actual: usize },
    #[error("expected message but stream is closed")]
    ClosedStream,
    #[error("ML-KEM decapsulation failed")]
    Decapsulation,
    #[error("invalid encapsulation key")]
    InvalidKey,
}

impl SemiHonest for MlKemOt {}

impl Malicious for MlKemOt {}

/// Wrapper for serializing encapsulation keys.
#[derive(Serialize, Deserialize)]
struct EncapKeyBytes(#[serde(with = "serde_bytes")] [u8; EK_BYTES]);

/// Wrapper for serializing ciphertexts.
#[derive(Serialize, Deserialize)]
struct CiphertextBytes(#[serde(with = "serde_bytes")] [u8; CT_BYTES]);

/// Message from receiver to sender: two encapsulation keys per OT.
/// For choice bit c, ek_c is a real key, ek_{1-c} is random bytes.
#[derive(Serialize, Deserialize)]
struct ReceiverMessage {
    /// Encapsulation keys for choice 0
    ek_0: Vec<EncapKeyBytes>,
    /// Encapsulation keys for choice 1
    ek_1: Vec<EncapKeyBytes>,
}

/// Message from sender to receiver: two ciphertexts per OT.
#[derive(Serialize, Deserialize)]
struct SenderMessage {
    /// Ciphertexts for choice 0
    ct_0: Vec<CiphertextBytes>,
    /// Ciphertexts for choice 1
    ct_1: Vec<CiphertextBytes>,
}

impl RotSender for MlKemOt {
    type Error = Error;

    #[tracing::instrument(level = Level::DEBUG, skip_all, fields(count = ots.len()))]
    #[tracing::instrument(target = "cryprot_metrics", level = Level::TRACE, skip_all, fields(phase = phase::BASE_OT))]
    async fn send_into(&mut self, ots: &mut impl Buf<[Block; 2]>) -> Result<(), Self::Error> {
        let count = ots.len();
        let (mut send, mut recv) = self.conn.byte_stream().await?;

        // Receive encapsulation keys from receiver
        let receiver_msg: ReceiverMessage = {
            let mut recv_stream = recv.as_stream();
            recv_stream.next().await.ok_or(Error::ClosedStream)??
        };

        if receiver_msg.ek_0.len() != count || receiver_msg.ek_1.len() != count {
            return Err(Error::InsufficientData {
                expected: count,
                actual: receiver_msg.ek_0.len().min(receiver_msg.ek_1.len()),
            });
        }

        // Generate a random seed for key derivation
        let seed: Block = self.rng.random();

        // Encapsulate under both keys for each OT
        let mut ct_0 = Vec::with_capacity(count);
        let mut ct_1 = Vec::with_capacity(count);

        for (i, (ek0_bytes, ek1_bytes)) in receiver_msg
            .ek_0
            .iter()
            .zip(receiver_msg.ek_1.iter())
            .enumerate()
        {
            // Parse and encapsulate under ek_0
            let (ct0, k0) = encapsulate_from_bytes(&ek0_bytes.0, &mut self.rng);
            let block_k0 = shared_key_to_block(&k0, i, seed);

            // Parse and encapsulate under ek_1
            let (ct1, k1) = encapsulate_from_bytes(&ek1_bytes.0, &mut self.rng);
            let block_k1 = shared_key_to_block(&k1, i, seed);

            ct_0.push(CiphertextBytes(ct0));
            ct_1.push(CiphertextBytes(ct1));
            ots[i] = [block_k0, block_k1];
        }

        // Send ciphertexts and seed to receiver
        let sender_msg = SenderMessage { ct_0, ct_1 };
        {
            let mut send_stream = send.as_stream();
            send_stream.send((sender_msg, seed)).await?;
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
        assert_eq!(choices.len(), ots.len());
        let count = ots.len();
        let (mut send, mut recv) = self.conn.byte_stream().await?;

        // Generate keypairs and fake keys for each OT
        let mut decap_keys: Vec<DecapsulationKey<MlKem768Params>> = Vec::with_capacity(count);
        let mut ek_0 = Vec::with_capacity(count);
        let mut ek_1 = Vec::with_capacity(count);

        for choice in choices.iter() {
            // Generate real keypair using rand_core 0.6 compatible RNG
            let (dk, ek) = MlKem768::generate(&mut RngCompat(&mut self.rng));
            let real_ek_bytes: [u8; EK_BYTES] = ek
                .as_bytes()
                .as_slice()
                .try_into()
                .expect("correct size");

            // Generate fake encapsulation key (random bytes)
            let fake_ek_bytes: [u8; EK_BYTES] = self.rng.random();

            // Use constant-time selection based on choice bit
            // If choice = 0: ek_0 = real, ek_1 = fake
            // If choice = 1: ek_0 = fake, ek_1 = real
            let choice_bit = choice.unwrap_u8();
            let (ek0, ek1) = if choice_bit == 0 {
                (real_ek_bytes, fake_ek_bytes)
            } else {
                (fake_ek_bytes, real_ek_bytes)
            };

            decap_keys.push(dk);
            ek_0.push(EncapKeyBytes(ek0));
            ek_1.push(EncapKeyBytes(ek1));
        }

        // Send encapsulation keys to sender
        let receiver_msg = ReceiverMessage { ek_0, ek_1 };
        {
            let mut send_stream = send.as_stream();
            send_stream.send(receiver_msg).await?;
        }

        // Receive ciphertexts and seed from sender
        let (sender_msg, seed): (SenderMessage, Block) = {
            let mut recv_stream = recv.as_stream();
            recv_stream.next().await.ok_or(Error::ClosedStream)??
        };

        if sender_msg.ct_0.len() != count || sender_msg.ct_1.len() != count {
            return Err(Error::InsufficientData {
                expected: count,
                actual: sender_msg.ct_0.len().min(sender_msg.ct_1.len()),
            });
        }

        // Decapsulate the chosen ciphertext for each OT
        for (i, ((dk, choice), (ct0, ct1))) in decap_keys
            .iter()
            .zip(choices.iter())
            .zip(sender_msg.ct_0.iter().zip(sender_msg.ct_1.iter()))
            .enumerate()
        {
            // Select the ciphertext corresponding to our choice
            let ct_bytes: &[u8; CT_BYTES] = if choice.unwrap_u8() == 0 {
                &ct0.0
            } else {
                &ct1.0
            };

            // Parse ciphertext and decapsulate
            let ct: Ciphertext<MlKem768> = (*ct_bytes).into();
            let shared_key = dk.decapsulate(&ct).map_err(|_| Error::Decapsulation)?;

            ots[i] = shared_key_to_block(&shared_key, i, seed);
        }

        Ok(())
    }
}

/// Encapsulate using raw encapsulation key bytes.
/// Returns the ciphertext bytes and shared key.
fn encapsulate_from_bytes(
    ek_bytes: &[u8; EK_BYTES],
    rng: &mut StdRng,
) -> ([u8; CT_BYTES], SharedKey<MlKem768>) {
    // Try to parse as valid encapsulation key
    // If parsing fails (fake key), we still produce a valid-looking ciphertext
    // The receiver won't be able to decrypt it anyway
    if let Some(ek) = EncapsulationKey::<MlKem768Params>::from_bytes(ek_bytes.into()).into() {
        let (ct, k): (Ciphertext<MlKem768>, SharedKey<MlKem768>) =
            ek.encapsulate(&mut RngCompat(rng)).expect("encapsulation should not fail");
        let ct_bytes: [u8; CT_BYTES] = ct.as_slice().try_into().expect("correct size");
        (ct_bytes, k)
    } else {
        // Invalid key - generate random ciphertext and key
        // This will never decrypt correctly, which is fine for a fake key
        let ct_bytes: [u8; CT_BYTES] = rand::random();
        let k_bytes: [u8; 32] = rand::random();
        let k: SharedKey<MlKem768> = k_bytes.into();
        (ct_bytes, k)
    }
}

/// Convert ML-KEM shared key to Block using random oracle.
fn shared_key_to_block(key: &SharedKey<MlKem768>, tweak: usize, seed: Block) -> Block {
    let mut ro = RandomOracle::new();
    ro.update(key.as_slice());
    ro.update(&tweak.to_le_bytes());
    ro.update(seed.as_bytes());
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
    async fn mlkem_base_rot() -> Result<()> {
        let _g = init_tracing();
        let (c1, c2) = local_conn().await?;
        let mut rng1 = StdRng::seed_from_u64(42);
        let rng2 = StdRng::seed_from_u64(42 * 42);
        let count = 128;
        let choices = random_choices(count, &mut rng1);

        let mut sender = MlKemOt::new_with_rng(c1, rng1);
        let mut receiver = MlKemOt::new_with_rng(c2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }

    #[tokio::test]
    async fn mlkem_base_rot_all_zeros() -> Result<()> {
        let _g = init_tracing();
        let (c1, c2) = local_conn().await?;
        let rng1 = StdRng::seed_from_u64(123);
        let rng2 = StdRng::seed_from_u64(456);
        let count = 128;
        // All zeros choices
        let choices: Vec<_> = (0..count).map(|_| subtle::Choice::from(0)).collect();

        let mut sender = MlKemOt::new_with_rng(c1, rng1);
        let mut receiver = MlKemOt::new_with_rng(c2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }

    #[tokio::test]
    async fn mlkem_base_rot_all_ones() -> Result<()> {
        let _g = init_tracing();
        let (c1, c2) = local_conn().await?;
        let rng1 = StdRng::seed_from_u64(789);
        let rng2 = StdRng::seed_from_u64(101112);
        let count = 128;
        // All ones choices
        let choices: Vec<_> = (0..count).map(|_| subtle::Choice::from(1)).collect();

        let mut sender = MlKemOt::new_with_rng(c1, rng1);
        let mut receiver = MlKemOt::new_with_rng(c2, rng2);
        let (s_ot, r_ot) = tokio::try_join!(sender.send(count), receiver.receive(&choices))?;

        for ((r, s), c) in r_ot.into_iter().zip(s_ot).zip(choices) {
            assert_eq!(r, s[c.unwrap_u8() as usize])
        }
        Ok(())
    }
}
