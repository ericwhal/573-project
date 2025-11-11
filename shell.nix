#!/user/bin/env nix-shell
{ pkgs ? import<nixpkgs> { } }:
(
  pkgs.buildFHSEnv {
    name = "ece573-project-FHS";
    targetPkgs = pkgs: (with pkgs; [
      gcc glibc boost scons pre-commit protobuf protobufc gperftools hdf5 gnum4 capstone libpng libelf wget cmake doxygen clang-tools
      curl libmpc mpfr gmp texinfo gperf patchutils bc zlib expat libslirp flex bison
      python313 ] ++ (with python313Packages; [
        pydot
        mypy
        tkinter
        pkgconfig
        pip
	protobuf
      ])
    );
    runScript = "bash";
    extraOutputsToInstall = [ "dev" ];
  }
).env
