# Tailscale Funnel setup

The public endpoint is exposed via [Tailscale Funnel](https://tailscale.com/kb/1223/funnel) — it terminates TLS at Tailscale's edge and tunnels traffic to the VPS over the Tailscale network, so the origin IP is never on the open internet.

## One-time setup on the VPS

```bash
# 1. Install Tailscale (Debian 13 trixie shown)
curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.noarmor.gpg \
  | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.tailscale-keyring.list \
  | sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt-get update && sudo apt-get install -y tailscale
sudo systemctl enable --now tailscaled

# 2. Authenticate (opens a URL — visit it in your browser to add the device)
sudo tailscale up --hostname=antiplagiat-mcp \
                  --ssh=false \
                  --accept-routes=false \
                  --accept-dns=false

# 3. Enable Serve + Funnel for the tailnet
#    https://login.tailscale.com/admin/settings/features  (one-time toggle)

# 4. Expose the local backend over Funnel
sudo tailscale funnel --bg http://localhost:8765
```

The first request takes ~30 seconds while Tailscale provisions the Let's Encrypt cert; subsequent requests are immediate.

## Verify

```bash
tailscale funnel status
curl https://antiplagiat-mcp.<your-tailnet>.ts.net/healthz
```

## What happens when the VPS reboots

`tailscale serve` and `tailscale funnel` configurations are persistent — `tailscaled` reapplies them automatically on boot. Nothing extra to do.

## To take the public endpoint offline

```bash
sudo tailscale funnel --https=443 off
```

The local FastAPI app keeps running, but the Tailscale edge no longer accepts public traffic.
