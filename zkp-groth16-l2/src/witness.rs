use crate::{L2PublicInput, L2Witness, DEFAULT_DIFF_BITS};
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::One;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use std::fmt;

#[derive(Debug, Clone)]
pub struct WitnessBuildConfig {
    pub scale: f64,
    pub clip: Option<f64>,
    pub tau: f64,
    pub diff_bit_len: usize,
}

impl Default for WitnessBuildConfig {
    fn default() -> Self {
        Self {
            scale: 1e4,
            clip: None,
            tau: 1.0,
            diff_bit_len: DEFAULT_DIFF_BITS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WitnessBuildResult {
    pub witness: L2Witness,
    pub public_input: L2PublicInput,
    pub num_values: usize,
    pub l2_sq: BigUint,
    pub tau_sq: BigUint,
}

#[derive(Debug)]
pub enum WitnessError {
    InvalidScale,
    InvalidTau,
    InvalidDiffBits,
    QuantizationOverflow,
    TauExceeded { l2_sq: BigUint, tau_sq: BigUint },
    DiffBitsOverflow { needed_bits: u64, diff_bits: usize },
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WitnessError::InvalidScale => write!(f, "scale must be positive"),
            WitnessError::InvalidTau => write!(f, "tau must be positive"),
            WitnessError::InvalidDiffBits => {
                write!(f, "diff_bit_len must be greater than zero")
            }
            WitnessError::QuantizationOverflow => {
                write!(f, "value exceeds supported quantization range")
            }
            WitnessError::TauExceeded { .. } => write!(f, "L2 bound exceeded"),
            WitnessError::DiffBitsOverflow { needed_bits, diff_bits } => {
                write!(f, "diff bits insufficient: need {needed_bits}, have {diff_bits}")
            }
        }
    }
}

impl std::error::Error for WitnessError {}

pub fn build_witness_from_f32(
    values: &[f32],
    config: &WitnessBuildConfig,
) -> Result<WitnessBuildResult, WitnessError> {
    let quantized = quantize(values, config)?;
    build_witness_from_quantized(&quantized, config)
}

pub fn build_witness_from_quantized(
    values: &[i128],
    config: &WitnessBuildConfig,
) -> Result<WitnessBuildResult, WitnessError> {
    validate_config(config)?;
    let l2_sq = accumulate_l2(values);
    let tau_scaled = quantize_scalar(config.tau, config.scale)?;
    let tau_sq = square_bigint(&tau_scaled);
    if l2_sq > tau_sq {
        return Err(WitnessError::TauExceeded { l2_sq, tau_sq });
    }
    let diff = &tau_sq - &l2_sq;
    let needed_bits = diff.bits();
    if needed_bits > config.diff_bit_len as u64 {
        return Err(WitnessError::DiffBitsOverflow {
            needed_bits,
            diff_bits: config.diff_bit_len,
        });
    }

    let mut witness_values: Vec<Fr> = Vec::with_capacity(values.len());
    let mut hash = Fr::zero();
    for &val in values {
        let fr_val = fr_from_i128(val);
        hash += fr_val;
        witness_values.push(fr_val);
    }

    let mut diff_bits: Vec<Fr> = Vec::with_capacity(config.diff_bit_len);
    for i in 0..config.diff_bit_len {
        if diff.bit(i as u64) {
            diff_bits.push(Fr::one());
        } else {
            diff_bits.push(Fr::zero());
        }
    }

    let public_input = L2PublicInput {
        tau_squared: fr_from_biguint(&tau_sq),
        hash,
    };
    let witness = L2Witness {
        values: witness_values,
        diff_bits,
    };

    Ok(WitnessBuildResult {
        witness,
        public_input,
        num_values: values.len(),
        l2_sq,
        tau_sq,
    })
}

fn quantize(values: &[f32], config: &WitnessBuildConfig) -> Result<Vec<i128>, WitnessError> {
    validate_config(config)?;
    let mut out = Vec::with_capacity(values.len());
    for &val in values {
        let mut v = val as f64;
        if let Some(clip) = config.clip {
            if clip > 0.0 {
                v = v.clamp(-clip, clip);
            }
        }
        let scaled = (v * config.scale).round();
        if !scaled.is_finite()
            || scaled <= i128::MIN as f64
            || scaled >= i128::MAX as f64
        {
            return Err(WitnessError::QuantizationOverflow);
        }
        out.push(scaled as i128);
    }
    Ok(out)
}

fn validate_config(config: &WitnessBuildConfig) -> Result<(), WitnessError> {
    if config.scale <= 0.0 || !config.scale.is_finite() {
        return Err(WitnessError::InvalidScale);
    }
    if config.tau <= 0.0 || !config.tau.is_finite() {
        return Err(WitnessError::InvalidTau);
    }
    if config.diff_bit_len == 0 {
        return Err(WitnessError::InvalidDiffBits);
    }
    Ok(())
}

fn quantize_scalar(tau: f64, scale: f64) -> Result<BigInt, WitnessError> {
    let scaled = (tau * scale).round();
    if !scaled.is_finite()
        || scaled <= i128::MIN as f64
        || scaled >= i128::MAX as f64
    {
        return Err(WitnessError::QuantizationOverflow);
    }
    Ok(BigInt::from(scaled as i128))
}

fn accumulate_l2(values: &[i128]) -> BigUint {
    let mut acc = BigUint::zero();
    for &val in values {
        let big = BigInt::from(val);
        let square = &big * &big;
        let sq_u = square.to_biguint().expect("square must be non-negative");
        acc += sq_u;
    }
    acc
}

fn square_bigint(value: &BigInt) -> BigUint {
    let sq = value * value;
    sq.to_biguint().expect("square must be non-negative")
}

fn fr_from_biguint(value: &BigUint) -> Fr {
    let bytes = value.to_bytes_le();
    if bytes.is_empty() {
        Fr::zero()
    } else {
        Fr::from_le_bytes_mod_order(&bytes)
    }
}

fn fr_from_i128(value: i128) -> Fr {
    let big = BigInt::from(value);
    match big.sign() {
        Sign::NoSign => Fr::zero(),
        Sign::Plus => fr_from_biguint(&big.to_biguint().unwrap()),
        Sign::Minus => {
            let mag = (-big).to_biguint().unwrap();
            -fr_from_biguint(&mag)
        }
    }
}
