use std::{env, fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

use anyhow::{anyhow, Context, Result};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use zkp_groth16_l2::{
    build_witness_from_f32,
    load_proof, load_proving_key, load_public_input, load_verifying_key, load_witness, prove,
    save_proof, save_proving_key, save_public_input, save_verifying_key, save_witness, setup, verify,
    L2Circuit, DEFAULT_DIFF_BITS, WitnessBuildConfig,
};

#[derive(Debug)]
struct SetupArgs {
    len: usize,
    pk: PathBuf,
    vk: PathBuf,
    seed: Option<u64>,
}

#[derive(Debug)]
struct ProveArgs {
    pk: PathBuf,
    witness: PathBuf,
    public: PathBuf,
    proof: PathBuf,
    seed: Option<u64>,
}

#[derive(Debug)]
struct VerifyArgs {
    vk: PathBuf,
    public: PathBuf,
    proof: PathBuf,
}

#[derive(Debug)]
struct WitnessArgs {
    input: PathBuf,
    len: Option<usize>,
    scale: f64,
    tau: f64,
    clip: Option<f64>,
    diff_bits: usize,
    witness_out: PathBuf,
    public_out: PathBuf,
}

#[derive(Debug)]
enum Command {
    Setup(SetupArgs),
    Prove(ProveArgs),
    Verify(VerifyArgs),
    Witness(WitnessArgs),
}

fn parse_args() -> Result<Command> {
    let mut iter = env::args().skip(1);
    let cmd = iter
        .next()
        .ok_or_else(|| anyhow!("missing subcommand (setup/prove/verify)"))?;
    match cmd.as_str() {
        "setup" => {
            let mut len = None;
            let mut pk = None;
            let mut vk = None;
            let mut seed = None;
            while let Some(flag) = iter.next() {
                match flag.as_str() {
                    "--len" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--len requires a value"))?;
                        len = Some(value.parse()?);
                    }
                    "--pk" => {
                        pk = Some(PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--pk requires a path"))?,
                        ));
                    }
                    "--vk" => {
                        vk = Some(PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--vk requires a path"))?,
                        ));
                    }
                    "--seed" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--seed requires a value"))?;
                        seed = Some(value.parse()?);
                    }
                    other => return Err(anyhow!("unknown flag {other} for setup")),
                }
            }
            Ok(Command::Setup(SetupArgs {
                len: len.ok_or_else(|| anyhow!("--len is required"))?,
                pk: pk.ok_or_else(|| anyhow!("--pk is required"))?,
                vk: vk.ok_or_else(|| anyhow!("--vk is required"))?,
                seed,
            }))
        }
        "prove" => {
            let mut args = ProveArgs {
                pk: PathBuf::new(),
                witness: PathBuf::new(),
                public: PathBuf::new(),
                proof: PathBuf::new(),
                seed: None,
            };
            let mut seen = (false, false, false, false);
            while let Some(flag) = iter.next() {
                match flag.as_str() {
                    "--pk" => {
                        args.pk = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--pk requires a path"))?,
                        );
                        seen.0 = true;
                    }
                    "--witness" => {
                        args.witness = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--witness requires a path"))?,
                        );
                        seen.1 = true;
                    }
                    "--public" => {
                        args.public = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--public requires a path"))?,
                        );
                        seen.2 = true;
                    }
                    "--proof" => {
                        args.proof = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--proof requires a path"))?,
                        );
                        seen.3 = true;
                    }
                    "--seed" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--seed requires a value"))?;
                        args.seed = Some(value.parse()?);
                    }
                    other => return Err(anyhow!("unknown flag {other} for prove")),
                }
            }
            if !seen.0 || !seen.1 || !seen.2 || !seen.3 {
                return Err(anyhow!("--pk, --witness, --public, and --proof are required"));
            }
            Ok(Command::Prove(args))
        }
        "verify" => {
            let mut args = VerifyArgs {
                vk: PathBuf::new(),
                public: PathBuf::new(),
                proof: PathBuf::new(),
            };
            let mut seen = (false, false, false);
            while let Some(flag) = iter.next() {
                match flag.as_str() {
                    "--vk" => {
                        args.vk = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--vk requires a path"))?,
                        );
                        seen.0 = true;
                    }
                    "--public" => {
                        args.public = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--public requires a path"))?,
                        );
                        seen.1 = true;
                    }
                    "--proof" => {
                        args.proof = PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--proof requires a path"))?,
                        );
                        seen.2 = true;
                    }
                    other => return Err(anyhow!("unknown flag {other} for verify")),
                }
            }
            if !seen.0 || !seen.1 || !seen.2 {
                return Err(anyhow!("--vk, --public, and --proof are required"));
            }
            Ok(Command::Verify(args))
        }
        "witness" => {
            let mut input = None;
            let mut len = None;
            let mut scale = None;
            let mut tau = None;
            let mut clip = None;
            let mut diff_bits = DEFAULT_DIFF_BITS;
            let mut witness_out = None;
            let mut public_out = None;
            while let Some(flag) = iter.next() {
                match flag.as_str() {
                    "--input" => {
                        input = Some(PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--input requires a path"))?,
                        ));
                    }
                    "--len" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--len requires a value"))?;
                        len = Some(value.parse()?);
                    }
                    "--scale" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--scale requires a value"))?;
                        scale = Some(value.parse()?);
                    }
                    "--tau" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--tau requires a value"))?;
                        tau = Some(value.parse()?);
                    }
                    "--clip" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--clip requires a value"))?;
                        clip = Some(value.parse()?);
                    }
                    "--diff-bits" => {
                        let value = iter
                            .next()
                            .ok_or_else(|| anyhow!("--diff-bits requires a value"))?;
                        diff_bits = value.parse()?;
                    }
                    "--witness-out" => {
                        witness_out = Some(PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--witness-out requires a path"))?,
                        ));
                    }
                    "--public-out" => {
                        public_out = Some(PathBuf::from(
                            iter.next().ok_or_else(|| anyhow!("--public-out requires a path"))?,
                        ));
                    }
                    other => return Err(anyhow!("unknown flag {other} for witness")),
                }
            }
            Ok(Command::Witness(WitnessArgs {
                input: input.ok_or_else(|| anyhow!("--input is required"))?,
                len,
                scale: scale.ok_or_else(|| anyhow!("--scale is required"))?,
                tau: tau.ok_or_else(|| anyhow!("--tau is required"))?,
                clip,
                diff_bits,
                witness_out: witness_out.ok_or_else(|| anyhow!("--witness-out is required"))?,
                public_out: public_out.ok_or_else(|| anyhow!("--public-out is required"))?,
            }))
        }
        other => Err(anyhow!("unknown subcommand {other}")),
    }
}

fn build_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let derived = (nanos & u128::from(u64::MAX)) as u64;
            StdRng::seed_from_u64(derived)
        },
    }
}

fn read_f32_file(path: &PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "input byte length {} is not divisible by 4", bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("chunk size checked");
        values.push(f32::from_le_bytes(arr));
    }
    Ok(values)
}

fn main() -> Result<()> {
    match parse_args()? {
        Command::Setup(args) => {
            let mut rng = build_rng(args.seed);
            let (pk, vk) = setup(&mut rng, args.len)?;
            save_proving_key(&args.pk, &pk).context("writing proving key")?;
            save_verifying_key(&args.vk, &vk).context("writing verifying key")?;
        }
        Command::Prove(args) => {
            let pk = load_proving_key(&args.pk).context("loading proving key")?;
            let witness = load_witness(&args.witness).context("loading witness")?;
            let public = load_public_input(&args.public).context("loading public input")?;
            let circuit = L2Circuit {
                public_input: public,
                witness,
                diff_bit_len: DEFAULT_DIFF_BITS,
            };
            let mut rng = build_rng(args.seed);
            let proof = prove(&pk, circuit, &mut rng)?;
            save_proof(&args.proof, &proof).context("writing proof")?;
        }
        Command::Verify(args) => {
            let vk = load_verifying_key(&args.vk).context("loading verifying key")?;
            let proof = load_proof(&args.proof).context("loading proof")?;
            let public = load_public_input(&args.public).context("loading public input")?;
            let ok = verify(&vk, &proof, &public)?;
            if !ok {
                return Err(anyhow!("verification failed"));
            }
        }
        Command::Witness(args) => {
            let floats = read_f32_file(&args.input)?;
            if let Some(expected) = args.len {
                if expected != floats.len() {
                    return Err(anyhow!(
                        "input length mismatch: expected {expected}, got {}",
                        floats.len()
                    ));
                }
            }
            let config = WitnessBuildConfig {
                scale: args.scale,
                clip: args.clip,
                tau: args.tau,
                diff_bit_len: args.diff_bits.max(1),
            };
            let result = build_witness_from_f32(&floats, &config)?;
            save_witness(&args.witness_out, &result.witness).context("writing witness")?;
            save_public_input(&args.public_out, &result.public_input)
                .context("writing public input")?;
            println!(
                "witness prepared: samples={} l2_sq={} tau_sq={}",
                result.num_values,
                result.l2_sq.to_str_radix(10),
                result.tau_sq.to_str_radix(10)
            );
        }
    }
    Ok(())
}
