use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    // Skip MOE kernel build for broader GPU support (SM75, SM60, etc.)
    // MOE kernels use bfloat16 WMMA which requires SM80+
    // Qwen3-ASR doesn't use MOE, so this is safe to skip
    println!("cargo:warning=Skipping MOE kernel build for broader GPU compatibility");

    // Check if we're on Windows (MSVC) or Unix
    let is_target_msvc = if let Ok(target) = std::env::var("TARGET") {
        target.contains("msvc")
    } else {
        false
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if is_target_msvc {
        // On Windows MSVC, create a dummy empty object file and static library
        let obj_path = out_dir.join("moe_dummy.obj");
        let lib_path = out_dir.join("moe.lib");

        // Create minimal COFF object file (empty)
        // This is a minimal valid COFF format for x64
        let coff_header: [u8; 20] = [
            0x64, 0x86, // Machine: x64
            0x00, 0x00, // NumberOfSections: 0
            0x00, 0x00, 0x00, 0x00, // TimeDateStamp
            0x00, 0x00, 0x00, 0x00, // PointerToSymbolTable
            0x00, 0x00, 0x00, 0x00, // NumberOfSymbols
            0x00, 0x00, // SizeOfOptionalHeader
            0x00, 0x00, // Characteristics
        ];
        std::fs::write(&obj_path, coff_header).expect("Failed to write dummy obj");

        // Use lib.exe to create the static library
        let lib_result = std::process::Command::new("lib")
            .arg("/NOLOGO")
            .arg("/OUT:")
            .arg(&lib_path)
            .arg(&obj_path)
            .status();

        match lib_result {
            Ok(status) if status.success() => {
                println!("cargo:rustc-link-search={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=moe");
            }
            _ => {
                // Fallback: try with full path to lib.exe
                let lib_exe = std::env::var("LIBEXE").unwrap_or_else(|_| "lib.exe".to_string());
                let _ = std::process::Command::new(&lib_exe)
                    .arg("/NOLOGO")
                    .arg("/OUT:")
                    .arg(&lib_path)
                    .arg(&obj_path)
                    .status();

                // Create minimal lib file manually if lib.exe failed
                if !lib_path.exists() {
                    // Minimal archive format
                    let archive_content = b"!<arch>\n";
                    std::fs::write(&lib_path, archive_content).ok();
                }

                println!("cargo:rustc-link-search={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=moe");
            }
        }
    } else {
        // On Unix, create an empty archive
        let lib_path = out_dir.join("libmoe.a");
        let _ = std::process::Command::new("ar")
            .arg("rcs")
            .arg(&lib_path)
            .status();

        if !lib_path.exists() {
            // Minimal archive format
            let archive_content = b"!<arch>\n";
            std::fs::write(&lib_path, archive_content).ok();
        }

        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=moe");
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
