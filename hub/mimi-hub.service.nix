  # Paste into ~/nix-config/home/server.nix (e.g. after the wolfram-multiplexer service), then `nrs`.
  # Replaces hub/run.sh: stop that one first with `systemctl --user stop mimi-hub` (same unit name).
  #
  # mimi-hub — on-demand podcast conversion to .mimi (Mimi neural codec, ~2 MB/hour) plus Parakeet
  # transcripts with speaker labels, for the patched Pocket Casts and Mimi Bridge on the Fold 8.
  # Code + uv env live in ~/Lab_notebook/music/neural_audio (its own jj repo); nix only owns the service.
  # Tailnet-only: binds the server's Tailscale IP, port 8765 (tailscale0 is trusted, no firewall entry);
  # restarts until tailscaled has the address after boot.
  systemd.user.services.mimi-hub = {
    Unit = {
      Description = "mimi-hub (podcast -> .mimi + transcripts)";
      After = [ "network-online.target" ];
    };
    Service = {
      WorkingDirectory = "${config.home.homeDirectory}/Lab_notebook/music/neural_audio";
      ExecStart = "${pkgs.uv}/bin/uv run uvicorn hub.mimi_hub:app --host <tailscale-ip> --port 8765";
      Environment = [
        "PATH=${lib.makeBinPath [ pkgs.ffmpeg pkgs.uv pkgs.coreutils ]}"
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
        "MIMI_HUB_THREADS=16"
      ];
      Restart = "on-failure";
      RestartSec = 10;
      MemoryMax = "24G";
      MemorySwapMax = "0";
    };
    Install.WantedBy = [ "default.target" ];
  };
