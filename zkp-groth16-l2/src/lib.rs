use ark_bn254::{Bn254, Fr};
use ark_ff::Field;
use ark_groth16::{prepare_verifying_key, Groth16, Proof, ProvingKey, VerifyingKey};
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::RngCore, Zero};
use std::fs::File;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

mod witness;
pub use witness::{
    build_witness_from_f32,
    build_witness_from_quantized,
    WitnessBuildConfig,
    WitnessBuildResult,
    WitnessError,
};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct L2PublicInput {
    /// tau^2 in Fr
    pub tau_squared: Fr,
    /// simple “hash” = sum of original values (placeholder; replace with Poseidon later)
    pub hash: Fr,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct L2Witness {
    /// fixed-point vector (already scaled/clipped)
    pub values: Vec<Fr>,
    /// Binary decomposition of tau_sq - sum(values^2)
    pub diff_bits: Vec<Fr>,
}

pub struct L2Circuit {
    pub public_input: L2PublicInput,
    pub witness: L2Witness,
    pub diff_bit_len: usize,
}

pub const DEFAULT_DIFF_BITS: usize = 256;

impl ConstraintSynthesizer<Fr> for L2Circuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        // public inputs
        let tau_sq_var = cs.new_input_variable(|| Ok(self.public_input.tau_squared))?;
        let hash_var = cs.new_input_variable(|| Ok(self.public_input.hash))?;

        // running sum of squares
        let mut running_sum_val = Fr::zero();
        let mut running_sum_var = cs.new_witness_variable(|| Ok(running_sum_val))?;

        // running hash (here：简单相加，可将来换成 Poseidon)
        let mut hash_acc_val = Fr::zero();
        let mut hash_acc_var = cs.new_witness_variable(|| Ok(hash_acc_val))?;

        for value in self.witness.values.iter() {
            let val_var = cs.new_witness_variable(|| Ok(*value))?;

            // square = v * v
            let square_val = *value * *value;
            let square_var = cs.new_witness_variable(|| Ok(square_val))?;
            cs.enforce_constraint(
                lc!() + val_var,
                lc!() + val_var,
                lc!() + square_var,
            )?;

            // update running sum of squares
            let new_sum_val = running_sum_val + square_val;
            let new_sum_var = cs.new_witness_variable(|| Ok(new_sum_val))?;
            cs.enforce_constraint(
                lc!() + running_sum_var + square_var,
                lc!() + Variable::One,
                lc!() + new_sum_var,
            )?;
            running_sum_val = new_sum_val;
            running_sum_var = new_sum_var;

            // update running “hash” = sum(values)
            let new_hash_val = hash_acc_val + *value;
            let new_hash_var = cs.new_witness_variable(|| Ok(new_hash_val))?;
            cs.enforce_constraint(
                lc!() + hash_acc_var + val_var,
                lc!() + Variable::One,
                lc!() + new_hash_var,
            )?;
            hash_acc_val = new_hash_val;
            hash_acc_var = new_hash_var;
        }

        if self.witness.diff_bits.len() != self.diff_bit_len {
            return Err(SynthesisError::AssignmentMissing);
        }

        // encode (tau^2 - sum_sq) as binary decomposition to ensure non-negative diff
        let mut diff_lc = lc!();
        let mut coeff = Fr::from(1u64);
        for bit in self.witness.diff_bits.iter() {
            let bit_var = cs.new_witness_variable(|| Ok(*bit))?;
            // Boolean constraint: bit * (bit - 1) = 0
            cs.enforce_constraint(
                lc!() + bit_var,
                lc!() + bit_var - Variable::One,
                lc!(),
            )?;
            diff_lc = diff_lc + (coeff, bit_var);
            coeff.double_in_place();
        }
        cs.enforce_constraint(
            diff_lc,
            lc!() + Variable::One,
            lc!() + tau_sq_var - running_sum_var,
        )?;

        // enforce hash consistency
        cs.enforce_constraint(
            lc!() + hash_acc_var,
            lc!() + Variable::One,
            lc!() + hash_var,
        )?;

        Ok(())
    }
}

pub fn setup<R: RngCore>(
    rng: &mut R,
    len: usize,
) -> Result<(ProvingKey<Bn254>, VerifyingKey<Bn254>), SynthesisError> {
    let dummy_input = L2PublicInput {
        tau_squared: Fr::zero(),
        hash: Fr::zero(),
    };
    let dummy_witness = L2Witness {
        values: vec![Fr::zero(); len],
        diff_bits: vec![Fr::zero(); DEFAULT_DIFF_BITS],
    };
    let circuit = L2Circuit {
        public_input: dummy_input,
        witness: dummy_witness,
        diff_bit_len: DEFAULT_DIFF_BITS,
    };
    let pk = Groth16::<Bn254>::generate_random_parameters_with_reduction(circuit, rng)?;
    let vk = pk.vk.clone();
    Ok((pk, vk))
}

pub fn prove(
    pk: &ProvingKey<Bn254>,
    circuit: L2Circuit,
    rng: &mut impl RngCore,
) -> Result<Proof<Bn254>, SynthesisError> {
    Groth16::<Bn254>::create_random_proof_with_reduction(circuit, pk, rng)
}

pub fn verify(
    vk: &VerifyingKey<Bn254>,
    proof: &Proof<Bn254>,
    public_input: &L2PublicInput,
) -> Result<bool, SynthesisError> {
    let pvk = prepare_verifying_key(vk);
    let inputs = vec![public_input.tau_squared, public_input.hash];
    Groth16::<Bn254>::verify_proof(&pvk, proof, &inputs)
}

// --------- helper IO routines ---------

fn ark_err(e: ark_serialize::SerializationError) -> IoError {
    IoError::new(ErrorKind::Other, e)
}

pub fn save_proving_key(path: impl AsRef<Path>, pk: &ProvingKey<Bn254>) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    pk.serialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn load_proving_key(path: impl AsRef<Path>) -> std::io::Result<ProvingKey<Bn254>> {
    let mut f = File::open(path)?;
    ProvingKey::deserialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn save_verifying_key(path: impl AsRef<Path>, vk: &VerifyingKey<Bn254>) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    vk.serialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn load_verifying_key(path: impl AsRef<Path>) -> std::io::Result<VerifyingKey<Bn254>> {
    let mut f = File::open(path)?;
    VerifyingKey::deserialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn save_proof(path: impl AsRef<Path>, proof: &Proof<Bn254>) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    proof.serialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn load_proof(path: impl AsRef<Path>) -> std::io::Result<Proof<Bn254>> {
    let mut f = File::open(path)?;
    Proof::deserialize_uncompressed(&mut f).map_err(ark_err)
}

pub fn save_public_input(path: impl AsRef<Path>, public: &L2PublicInput) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    public.serialize_compressed(&mut f).map_err(ark_err)
}

pub fn load_public_input(path: impl AsRef<Path>) -> std::io::Result<L2PublicInput> {
    let mut f = File::open(path)?;
    L2PublicInput::deserialize_compressed(&mut f).map_err(ark_err)
}

pub fn save_witness(path: impl AsRef<Path>, witness: &L2Witness) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    witness.serialize_compressed(&mut f).map_err(ark_err)
}

pub fn load_witness(path: impl AsRef<Path>) -> std::io::Result<L2Witness> {
    let mut f = File::open(path)?;
    L2Witness::deserialize_compressed(&mut f).map_err(ark_err)
}
