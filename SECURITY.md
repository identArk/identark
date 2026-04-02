# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the IdentArk SDK, please report it responsibly by emailing security@identark.dev instead of using the public issue tracker.

### Responsible Disclosure Guidelines

1. **Do not publicly disclose the vulnerability** until we have had time to address it
2. **Provide detailed information** including:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any proof-of-concept code (if applicable)
3. **Allow time for a fix** before any public disclosure

### Response Timeline

- **48 hours**: We will acknowledge receipt of your security report
- **7 days**: We will triage the vulnerability and determine severity
- **30 days**: Target for releasing a security patch (may vary based on complexity)
- **Full disclosure**: Coordinated disclosure after patch is released

## Security Contact

- **Email**: security@identark.dev
- **Response Time**: We aim to respond to security reports within 48 hours

## Security Best Practices

When using the IdentArk SDK:

1. Keep the SDK updated to the latest version
2. Review the CHANGELOG for security patches
3. Use strong authentication credentials
4. Never commit credentials to version control
5. Use environment variables for sensitive configuration
6. Enable Dependabot to stay notified of dependency updates

## Automated Security Scanning

This project uses:

- **CodeQL**: Static analysis for code security vulnerabilities
- **Dependabot**: Automated dependency updates and vulnerability alerts
- **OSSF Scorecard**: Supply chain security assessment

Security results are available in the GitHub Security tab.
