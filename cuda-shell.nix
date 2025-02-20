# Run with `nix-shell cuda-shell.nix`
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
   allowUnfree = true;
   name = "cuda-env-shell";
   shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
   '';
}