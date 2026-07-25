# Release process

Publishing is disabled while the project contains only the scaffold status
probe. Before enabling publication:

1. assign the permanent repository URL, owners, and security contact;
2. replace the status-only surface with a useful validated numerical API;
3. complete the relevant source-map rows and migration review;
4. run formatting, strict Clippy, debug/release tests, MSRV, rustdoc,
   coverage, dependency, and Python wheel gates;
5. inspect `cargo package -p pykep-core --list`, wheel, and sdist contents;
6. install the crate and wheel into separate empty projects;
7. verify version agreement, license, provenance, documentation, and enabled
   features;
8. publish with trusted OIDC publishing only from a green protected release
   environment;
9. verify registry pages and docs before creating the GitHub release.

Never reuse a published version, commit a registry token, or tag a build that
has not passed the clean-environment smoke tests.
