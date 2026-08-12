<!--
PR title format: [Type] Concise imperative summary

Allowed type tags:
[Bugfix] [Feature] [Docs] [Refactor] [Perf] [Test] [CI/Release]

An optional subsystem tag may follow the type:
[Bugfix][SCOUT] Reject an inconclusive replay precondition
-->

## Summary

Describe the problem and the outcome of this change.

## Related issues

Link related issues with `Closes #...` or explain why no issue is needed.

## Changes

Describe the implementation and any user-visible behavior.

## Compatibility

Describe changes to public APIs, checkpoint formats, framework versions, distributed topology, or manager contracts.
Write `No compatibility impact` when none apply.

## Validation

List the exact commands and results.

```text
Command:
Result:
```

For distributed or GPU changes, include the framework, GPU model, host count, world size, topology, injected failure if any, and observed result.
If suitable hardware was unavailable, state which validation remains outstanding.

## Documentation

List updated documentation and examples, or explain why no documentation change is required.

## Checklist

- [ ] The pull request is focused on one coherent change.
- [ ] I added or updated tests for changed behavior, or explained why tests are not required.
- [ ] I ran the relevant CPU checks from `CONTRIBUTING.md`.
- [ ] I ran relevant distributed or GPU validation, or documented why it was not required or available.
- [ ] I updated public documentation, examples, and compatibility notes, or explained why they are not required.
- [ ] I preserved stable API compatibility or documented the compatibility impact.
- [ ] I removed credentials, private data, generated environments, and local validation artifacts.
- [ ] I identified copied or adapted code and preserved required attribution and licensing.
