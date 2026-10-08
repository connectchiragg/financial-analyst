# EC2 preparation and operation

This moves the existing bounded browser service to an always-running EC2 machine. It runs real read-only SQLite analytics and the exact provider/model selected in the private session configuration. There is no inference fallback. Preparation authenticates source PDFs and SQLite records, without calling a model. It does not create an instance, deploy files, or establish production reliability.

## Prepare the private directory locally

From the project root with its dependencies installed:

```sh
.venv/bin/python scripts/prepare_ec2_bundle.py --config .local/openrouter-config.json
```

The new `.local/ec2-bundle/` directory contains application code, pinned requirements, deployment templates, and a private artifact directory. The source catalog and session configuration use relative paths. Source PDFs, original proof pack, evidence, periods, units, scopes and status are preserved. SQLite is copied using its backup API, including committed WAL data. Only the selected provider credential is copied from the chosen environment file. The directory and every file are owner-readable only; do not commit, email, publicly upload, or paste their contents into logs.

Preparation refuses existing output directories. Use a new `--output .local/<name>` for a later release. Keep that release until EC2 verification succeeds so rollback is possible.

## Target and private transfer

Use a Linux EC2 instance with Python 3.10 or newer, `python3-venv`, systemd, and cloudflared installed at `/usr/bin/cloudflared`. Install cloudflared from [Cloudflare's official distribution instructions](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/). No GPU or local embedding model is required for the current provider-driven retrieval flow. Instance size and real memory/latency still require validation on the chosen machine.

The security group needs no public application ingress: the app binds to `127.0.0.1:8765` and the tunnel connects outbound. Use an authorized SSH source or Systems Manager for administration, and allow outbound HTTPS/model connectivity and Cloudflare tunnel connectivity according to [Cloudflare's firewall guidance](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/deploy-tunnels/tunnel-with-firewall/). Keep the EC2 volume encrypted and restrict instance administration.

Copy the bundle over authenticated SSH/SCP or another private transfer mechanism. Do not publish the artifact to the public repository or a public bucket. The target instructions below assume it was placed at `/home/ubuntu/ec2-bundle` and the instance is a fresh install; adapt only the incoming path for a different administrator user. Do not overwrite an active `/opt/financial-analyst` release without retaining its rollback copy.

## Install the first release on EC2

The privileged commands only install files and a service user; application and tunnel processes run as that non-root user.

```sh
sudo sh -eu <<'INSTALL'
# mkdir fails if an existing release is present; no files are overwritten.
mkdir -m 0700 /opt/financial-analyst
if ! id -u financial-analyst >/dev/null 2>&1; then
  useradd --system --user-group --home-dir /nonexistent --shell /usr/sbin/nologin financial-analyst
fi
cp -R /home/ubuntu/ec2-bundle/. /opt/financial-analyst/
chown -R financial-analyst:financial-analyst /opt/financial-analyst
find /opt/financial-analyst -type d -exec chmod 0700 {} +
find /opt/financial-analyst -type f -exec chmod 0600 {} +
INSTALL
sudo -u financial-analyst python3 -m venv /opt/financial-analyst/venv
sudo -u financial-analyst /opt/financial-analyst/venv/bin/python -m pip install -r /opt/financial-analyst/app/requirements.txt
sudo install -m 0644 /opt/financial-analyst/deploy/financial-analyst.service /etc/systemd/system/financial-analyst.service
sudo install -m 0644 /opt/financial-analyst/deploy/financial-analyst-tunnel.service /etc/systemd/system/financial-analyst-tunnel.service
sudo systemctl daemon-reload
sudo systemctl enable --now financial-analyst.service
curl --fail http://127.0.0.1:8765/health
sudo systemctl enable --now financial-analyst-tunnel.service
sudo journalctl -u financial-analyst-tunnel.service --since '5 minutes ago' --no-pager
```

The tunnel log shows the new HTTPS URL. Share that URL and the private access code through an authorized private channel. Do not print environment files to find the code.

Quick tunnels have no uptime guarantee and their hostname changes every time a new tunnel is created. EC2 removes the laptop dependency; it does not make this temporary hostname stable. For a stable URL, provide a domain and configure a named tunnel later. See [Cloudflare's current quick-tunnel limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## Acceptance before sharing

Check the public page, unauthorized `/ask` rejection, and at least one authenticated factual answer, one calculation, one qualitative answer, and one genuine refusal through the public URL. Inspect the exact source values, context, calculation and bottom citations. Use the real configured inference provider, rather than fixture responses. Verify both services recover after an EC2 restart, obtain the changed tunnel URL, and repeat an authenticated answer. Confirm the public URL remains usable when the laptop server and tunnel are stopped. A successful health response alone does not verify model access, answers, or source integrity.

Only after those checks pass should the old laptop link be replaced. Record the exact deployed code revision, configured provider/model, test results and new URL in private progress notes. This service remains one active answer request at a time; it is not a high-volume production deployment.
