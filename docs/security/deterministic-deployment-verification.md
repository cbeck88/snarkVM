# Non-deterministic deployment verification (ambient-RNG burner + `call.dynamic`)

## Summary

`Stack::verify_deployment` (`synthesizer/process/src/stack/deploy.rs`) is a consensus-critical
function: every validator must reach the **same** accept/reject decision on a byte-identical
deployment transaction. The function already goes to some length to be deterministic — it seeds a
`ChaChaRng` from the deployment ID and uses it to sample function inputs and the per-function
synthesis RNGs, with the comment:

> This is needed to ensure that the verification results of deployments are consistent across all
> parties, because currently there is a possible flakiness due to overflows in Field to Scalar
> casting.

However, two randomness sources in the same function were **not** switched to the seeded RNG and
continued to draw from the ambient `rng` argument:

1. the per-function **burner private key** (`let burner_private_key = PrivateKey::new(rng)?;`), and
2. the **record-translation** sampling used to verify translation certificates.

snarkOS passes a fresh, non-deterministic `rand::rng()` into this path (block validation, block
derivation, and `check_transaction_basic`), so each validator draws a different burner.

The burner sets `self.signer` for the dummy `CheckDeployment` synthesis. The `call.dynamic` opcode
lets a program choose its dynamic-call target at runtime and branch it on `self.signer`. In dummy
(`CheckDeployment`) mode, the target resolver treats the two possible targets asymmetrically
(`synthesizer/process/src/stack/call/dynamic.rs`, `resolve_dynamic_target`):

```rust
// Verify that the function is not a closure.
if substack.program().get_closure(&function_name).is_ok() {
    Err(anyhow!("Cannot dynamically evaluate a closure: {function_name}")) // hard error, even in dummy mode
} else if substack.program().contains_function(&function_name) {
    Ok(Some(...))
} else if in_dummy_mode {
    Ok(None) // a missing function is tolerated
} else {
    Err(anyhow!("Dynamic call to '{program_id}/{function_name}' is invalid or unsupported."))
}
```

Selecting a **closure** hard-errors; selecting a **missing function** is tolerated. A program that
derives one bit from `self.signer` and uses it to pick between a closure and a missing function
therefore verifies differently depending only on the ambient RNG the caller happens to pass in.

## Impact

The same correctly-signed deployment transaction is **accepted by some honest validators and
aborted by others**. From an identical certified subDAG, validators derive different blocks
(different transactions root, block hash, and installed-program set). This is a permanent chain
split; at large committee scale a ~50/50 split instead halts confirmation (neither branch reaches
`2f+1`). Preconditions are minimal: a single unprivileged funded account deploying one cheap
program via the normal public path.

## How this happened (development history)

This is a latent flaw that lay dormant for years until an unrelated feature made it exploitable.

| Date | Commit | Event |
|------|--------|-------|
| 2023-02-02 | `3c98e12ed` (raychu86) — *Add dedicated checks for deployment* | Deployment certificate checks introduced; the dummy **burner private key is sampled from the ambient `rng`**. Harmless at the time: no synthesis control flow depended on `self.signer`. |
| 2024-08-23 | `88a0d800d` (raychu86, PR #2535) — *Use a seeded rng for inputs and substack during deployment verification* | To fix Field→Scalar overflow flakiness, the input sampling and the per-function synthesis RNG seeds were switched to a **deployment-ID-seeded** `ChaChaRng`. Exactly three sites were converted; the **burner key and the `Request::sign` nonce were left on the ambient `rng`**, and were still harmless. |
| 2025-11-04 | `378934ddd` (Victor Sint Nicolaas) — *CallTrait for DynamicCall* | `resolve_dynamic_target` was added with the **closure check placed before the dummy-mode tolerance**, making synthesis-time control flow depend on the concrete ejected `self.signer`. This retroactively weaponized the 2024 residual into a consensus fork. |

In other words, PR #2535 seeded the RNG *precisely because* deployment verification must be
reproducible, but missed two of the randomness sources feeding the very same synthesis. (Note: the
`call.dynamic` opcode is enabled at `ConsensusVersion::V14`.)

## Fix

Draw **all** randomness in `verify_deployment` from the deployment-ID-seeded `seeded_rng`, so the
ambient `rng` argument no longer influences any verification output:

- the burner private key,
- the `Request::sign` nonce, and
- the record-translation sampling.

The burner is a dummy signer whose unpredictability was never a security property (the function
inputs are already deterministic), so making it deterministic is free and matches the original
intent of PR #2535. The ambient `rng` parameter is retained (renamed `_rng`) to preserve the public
signature.

## Regression test

`synthesizer/process/src/tests/test_deploy_rng_split.rs` deploys a program whose `call.dynamic`
target branches on `self.signer` (closure → hard error, missing function → tolerated), then verifies
the **same** deployment under 40 different ambient RNGs and asserts they all agree.

| | same deployment, 40 different ambient RNGs |
|---|---|
| Before the fix | 20 accepted, 20 rejected → chain fork |
| After the fix | 40 accepted, 0 rejected → deterministic |

Run:

```
cargo test -p snarkvm-synthesizer-process --features test \
  test_ambient_rng_burner_forks_deployment_verification -- --nocapture
```

## Related follow-up (separate change)

`get.record.dynamic` (`synthesizer/program/src/logic/instruction/operation/get_record_dynamic.rs`)
samples its dummy entry value from `rand::rng()` on the not-present branch, which is reached during
`CheckDeployment` synthesis. That sampled value is stored to a register and can likewise be fed into
a `call.dynamic` target, giving an **independent** non-determinism source not addressed by this
change. It is tracked and fixed separately.
