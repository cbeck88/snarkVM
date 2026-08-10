# Non-deterministic deployment verification via `get.record.dynamic`

## Summary

This is a second, **independent** instance of the non-deterministic-deployment-verification bug
class (the first being the ambient-RNG burner key in `Stack::verify_deployment`, fixed separately).
It is **not** closed by the burner fix.

`get.record.dynamic` extracts an entry from a dynamic record. Its helper `compute_or_sample_path`
(`synthesizer/program/src/logic/instruction/operation/get_record_dynamic.rs`) has a branch for when
the record data is not present:

```rust
None => {
    // Sample an arbitrary value for the entry, consistent with the specified type.
    let value = {
        let rng = &mut rand::rng();                 // <-- ambient, non-deterministic
        let address = Address::<N>::rand(rng);
        stack.sample_value(&address, &RegisterType::Plaintext(plaintext_type.clone()), rng)?
    };
    ...
}
```

That not-present branch is reached during `CheckDeployment` synthesis, i.e. during **deployment
verification**: a sampled dynamic-record input carries no data, because
`Stack::sample_dynamic_record` (`synthesizer/process/src/stack/helpers/sample.rs`) builds it with
`data = None`. So every `get.record.dynamic` executed against a sampled dynamic-record input during
verification hits the ambient-RNG sample.

The sampled value is stored to the destination register (as a private witness). A program can then
feed that register into a `call.dynamic` target operand. Exactly as with the burner vector, the
dynamic-call target resolver treats a **closure** (hard error) and a **missing function** (tolerated
in dummy mode) asymmetrically, so the concrete ejected value decides accept vs. reject. Because the
value is drawn from `rand::rng()`, each validator decides differently on the same byte-identical
deployment.

Crucially, the exploiting program never reads `self.signer`, so this splits validators **even when
the burner is drawn from a seeded RNG**. It is a distinct randomness source that must be fixed on its
own.

## Impact

Identical to the burner vector: the same correctly-signed deployment transaction is accepted by some
honest validators and aborted by others, forking the chain (or, at large committee scale, halting
confirmation). Preconditions are minimal — one unprivileged account deploying one cheap program that
contains a `get.record.dynamic` on a dynamic-record input plus a `call.dynamic` whose target is
chosen from the extracted value. (`get.record.dynamic` and `call.dynamic` are both enabled at
`ConsensusVersion::V14`.)

## Fix

Draw the dummy entry value from a **deterministic** RNG instead of `rand::rng()`. The RNG is seeded
from data that is identical across all validators verifying the same deployment — the record `root`
(which, in verification, is sampled from the deployment-ID-seeded RNG) combined with the entry
identifier:

```rust
let root_seed  = u64::from_bytes_le(&root.to_bytes_le()?[0..8])?;
let entry_seed = u64::from_bytes_le(&entry_identifier.to_field()?.to_bytes_le()?[0..8])?;
let rng = &mut ChaChaRng::seed_from_u64(root_seed ^ entry_seed);
```

The sampled value is a dummy witness used only for synthesis; only its determinism matters. For a
benign program (one that does not route the value into control flow) the value never affects circuit
structure, so verification is unchanged. For an adversarial program the value now selects the same
branch on every validator, closing the fork.

## Regression test

`synthesizer/process/src/tests/test_get_record_dynamic_rng_split.rs` deploys a program that reads an
entry via `get.record.dynamic` and branches a `call.dynamic` target on it (closure vs missing
function), then verifies the **same** deployment under 40 different ambient RNGs and asserts they all
agree. The program does not read `self.signer`, demonstrating independence from the burner vector.

| | same deployment, 40 different ambient RNGs |
|---|---|
| Before the fix | 18 accepted, 22 rejected → chain fork |
| After the fix | 40 accepted, 0 rejected → deterministic |

Run:

```
cargo test -p snarkvm-synthesizer-process --features test \
  test_get_record_dynamic_ambient_rng_forks_deployment_verification -- --nocapture
```
