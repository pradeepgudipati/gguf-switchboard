use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gguf-switchboard"))
        .args(args)
        .output()
        .expect("gguf-switchboard should run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_complete_help(text: &str) {
    for sample in [
        "ggs models search <query>",
        "ggs models search vllm <query>",
        "ggs models files <repo-id>",
        "ggs models pull <repo-id>",
        "ggs models pull vllm <repo-id>",
        "ggs discover-models <models-dir>",
        "ggs sync-hf-metadata",
        "ggs export-registry <models.toml>",
        "ggs status",
        "ggs stop",
        "ggs restart",
        "ggs logs",
        "ggs logs watch",
        "ggs logs --tail 250",
        "ggs <config.toml>",
    ] {
        assert!(
            text.contains(sample),
            "missing sample `{sample}` in:\n{text}"
        );
    }
}

#[test]
fn invalid_logs_arguments_show_correct_examples() {
    for args in [
        &["logs", "--tail"][..],
        &["logs", "--tail", "0"][..],
        &["logs", "--tail", "abc"][..],
        &["logs", "follow"][..],
    ] {
        let output = run(args);
        assert!(!output.status.success(), "unexpected success for {args:?}");

        let error = stderr(&output);
        assert!(error.contains("Examples:"), "{error}");
        assert!(error.contains("ggs logs"), "{error}");
        assert!(error.contains("ggs logs watch"), "{error}");
        assert!(error.contains("ggs logs --tail 100"), "{error}");
    }
}

#[test]
fn help_aliases_show_all_supported_commands() {
    for arg in ["help", "--help", "-h"] {
        let output = run(&[arg]);
        assert!(output.status.success(), "{arg} failed: {}", stderr(&output));
        assert_complete_help(&stdout(&output));
    }
}

#[test]
fn singular_model_suggests_models_and_preserves_the_command() {
    let output = run(&["model", "search", "vllm", "Muse"]);
    assert!(!output.status.success());

    let error = stderr(&output);
    assert!(
        !error.contains("\\n"),
        "help should render as real lines: {error}"
    );
    assert!(error.contains("Did you mean 'models'?"), "{error}");
    assert!(error.contains("ggs models search vllm Muse"), "{error}");
    assert_complete_help(&error);
    assert!(
        !error.contains("Failed to read config file 'model'"),
        "{error}"
    );
}

#[test]
fn unknown_top_level_command_shows_complete_help() {
    let output = run(&["modles"]);
    assert!(!output.status.success());

    let error = stderr(&output);
    assert!(error.contains("unknown command 'modles'"), "{error}");
    assert_complete_help(&error);
}

#[test]
fn model_command_mistakes_show_complete_help() {
    for args in [
        &["models", "unknown"][..],
        &["models", "files"][..],
        &["models", "search", "--unknown"][..],
    ] {
        let output = run(args);
        assert!(!output.status.success(), "unexpected success for {args:?}");
        assert_complete_help(&stderr(&output));
    }
}

#[test]
fn models_help_aliases_show_all_supported_commands() {
    for arg in ["help", "--help", "-h"] {
        let output = run(&["models", arg]);
        assert!(output.status.success(), "{arg} failed: {}", stderr(&output));
        assert_complete_help(&stdout(&output));
    }
}

#[test]
fn registry_command_mistakes_show_complete_help() {
    for (args, expected_error) in [
        (
            &["discover-models", "--unknown"][..],
            "discover-models: unknown flag '--unknown'",
        ),
        (
            &["sync-hf-metadata", "--output"][..],
            "sync-hf-metadata: missing value for --output",
        ),
        (
            &["export-registry"][..],
            "export-registry: missing input path",
        ),
    ] {
        let output = run(args);
        assert!(!output.status.success(), "unexpected success for {args:?}");
        let error = stderr(&output);
        assert!(error.contains(expected_error), "{error}");
        assert_complete_help(&error);
    }
}
