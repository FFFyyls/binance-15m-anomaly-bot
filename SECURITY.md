# Security Policy

## Reporting a vulnerability

Please do not publish credentials, account identifiers, order details, or an
exploit in a public issue. Use GitHub's private vulnerability reporting feature
for this repository. If private reporting is not enabled, open a minimal issue
requesting a private contact channel without including sensitive details.

## Trading safety

This software can submit real cryptocurrency futures orders. Treat every API
credential as a high-value secret:

- create separate testnet and production credentials;
- disable withdrawals and grant only the minimum trading permissions;
- use an IP allowlist where available;
- start with `DRY_RUN=1` and `USE_TESTNET=1`;
- rotate a credential immediately if it is committed, logged, or shared.

No security control in this repository is a substitute for exchange-side
permissions, account-level risk limits, or independent review.
