  # mimi-hub as a home-manager user service (the same unit as hub/mimi-hub.service). Paste into your home-manager
  # configuration and adjust the checkout path and the address; then rebuild.
  systemd.user.services.mimi-hub = {
    Unit = {
      Description = "mimi-hub (podcast -> .mimi + transcripts + ad marks)";
      After = [ "network-online.target" ];
    };
    Service = {
      WorkingDirectory = "${config.home.homeDirectory}/neural_audio";
      # a private address (e.g. your Tailscale IP); the app sends no token
      ExecStart = "${pkgs.bash}/bin/bash hub/serve.sh --host <private-ip> --port 8765";
      Environment = [
        # ffmpeg + uv; add the codex / claude CLIs' directory for ad detection
        "PATH=${lib.makeBinPath [ pkgs.ffmpeg pkgs.uv pkgs.coreutils ]}"
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
        "MIMI_HUB_THREADS=16"
      ];
      Restart = "on-failure";
      RestartSec = 10;  # retries until e.g. tailscaled has the address after boot
      MemoryMax = "24G";
    };
    Install.WantedBy = [ "default.target" ];
  };
