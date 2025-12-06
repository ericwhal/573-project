#!/user/bin/env nix-shell
{ pkgs ? import<nixpkgs> { } }: 
(
  pkgs.buildFHSEnv {
    name = "ece573-project-FHS";
    targetPkgs = pkgs: (with pkgs; [
      pkg-config
      dtc cdk #ncurses5
      gcc glibc boost scons pre-commit protobuf_21 protobufc gperftools hdf5 hdf5-cpp gnum4 capstone libpng libelf wget cmake doxygen clang-tools
      curl libmpc mpfr gmp texinfo gperf patchutils bc zlib expat libslirp flex bison
      python313 ] ++ (with python313Packages; [
        virtualenv
        pip
        pyyaml
        pydot
        mypy
        tkinter
        pkgconfig
        protobuf
	pandas
      ])
    );
    runScript = "bash";
    extraOutputsToInstall = [ "dev" ];
  }
).env
