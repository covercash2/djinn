//! Runs the real `djinn-agent` binary against a fake OpenAI-compatible server.

use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, post};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_djinn-agent");

fn sse(events: &[Value]) -> String {
    let mut body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
    body.push_str("data: [DONE]\n\n");
    body
}

fn text_reply(text: &str) -> String {
    sse(&[
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ])
}

fn tool_reply(name: &str, arguments: &str) -> String {
    sse(&[
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}]}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ])
}

type Replies = Arc<Mutex<Vec<String>>>;
type Requests = Arc<Mutex<Vec<Value>>>;

/// Serves the given replies in order, one per chat request, and records the requests.
async fn serve(replies: Vec<String>) -> (String, Requests) {
    let requests = Requests::default();
    let state = (Arc::new(Mutex::new(replies)) as Replies, requests.clone());
    async fn chat(State((replies, requests)): State<(Replies, Requests)>, body: Bytes) -> Response {
        requests
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&body).unwrap());
        let reply = {
            let mut replies = replies.lock().unwrap();
            if replies.is_empty() {
                String::from("data: [DONE]\n\n")
            } else {
                replies.remove(0)
            }
        };
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(reply))
            .unwrap()
    }
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(|| async { "ok" }))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}/v1"), requests)
}

/// A URL where nothing is listening.
fn dead_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}/v1")
}

async fn run(args: Vec<String>, stdin: &'static str) -> Output {
    tokio::task::spawn_blocking(move || {
        let mut child = Command::new(BIN)
            .args(args)
            .env_remove("DJINN_AGENT_BASE_URL")
            .env_remove("DJINN_AGENT_MODEL")
            .env_remove("DJINN_AGENT_API_KEY")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        {
            use std::io::Write;
            let mut input = child.stdin.take().unwrap();
            input.write_all(stdin.as_bytes()).unwrap();
        } // closing stdin ends the session
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[tokio::test]
async fn once_prints_the_answer_and_exits_zero() {
    let (url, requests) = serve(vec![text_reply("Paris.")]).await;
    let output = run(
        args(&[
            "--base-url",
            &url,
            "--model",
            "m",
            "--once",
            "Capital of France?",
        ]),
        "",
    )
    .await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), "Paris.\n");
    let requests = requests.lock().unwrap();
    assert_eq!(requests[0]["model"], "m");
    assert_eq!(requests[0]["messages"][1]["content"], "Capital of France?");
    let tools: Vec<_> = requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(tools, vec!["current_time", "calculator", "read_file"]);
}

#[tokio::test]
async fn once_runs_a_tool_and_keeps_tool_activity_off_stdout() {
    let (url, requests) = serve(vec![
        tool_reply("calculator", r#"{"expression":"6*7"}"#),
        text_reply("The answer is 42."),
    ])
    .await;
    let output = run(args(&["--base-url", &url, "--once", "What is 6*7?"]), "").await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), "The answer is 42.\n");
    assert!(
        text(&output.stderr).contains("[tool] calculator"),
        "{}",
        text(&output.stderr)
    );

    let requests = requests.lock().unwrap();
    let last = requests[1]["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(last["role"], "tool");
    assert_eq!(last["tool_call_id"], "call_1");
    assert_eq!(last["content"], "42");
}

#[tokio::test]
async fn once_with_an_unreachable_server_fails_naming_the_url() {
    let url = dead_url();
    let output = run(args(&["--base-url", &url, "--once", "hi"]), "").await;

    assert!(!output.status.success());
    assert_eq!(text(&output.stdout), "", "nothing on stdout on failure");
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains(&format!("{url}/chat/completions")),
        "{stderr}"
    );
}

#[tokio::test]
async fn the_step_limit_makes_once_fail() {
    let (url, _) = serve(vec![
        tool_reply("current_time", "{}"),
        tool_reply("current_time", "{}"),
        text_reply("never reached"),
    ])
    .await;
    let output = run(
        args(&["--base-url", &url, "--max-steps", "2", "--once", "loop"]),
        "",
    )
    .await;
    assert!(!output.status.success());
    assert_eq!(text(&output.stdout), "");
    assert!(
        text(&output.stderr).contains("step limit"),
        "{}",
        text(&output.stderr)
    );
}

#[tokio::test]
async fn a_missing_system_prompt_file_fails_before_contacting_the_server() {
    let (url, requests) = serve(vec![text_reply("x")]).await;
    let output = run(
        args(&[
            "--base-url",
            &url,
            "--system",
            "/no/such/prompt.txt",
            "--once",
            "hi",
        ]),
        "",
    )
    .await;

    assert_eq!(output.status.code(), Some(1));
    assert!(
        text(&output.stderr).contains("/no/such/prompt.txt"),
        "{}",
        text(&output.stderr)
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "the server must not be contacted"
    );
}

#[tokio::test]
async fn a_blank_system_prompt_file_fails_before_contacting_the_server() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blank.txt");
    std::fs::write(&path, "   \n").unwrap();
    let (url, requests) = serve(vec![text_reply("x")]).await;
    let output = run(
        args(&[
            "--base-url",
            &url,
            "--system",
            path.to_str().unwrap(),
            "--once",
            "hi",
        ]),
        "",
    )
    .await;

    assert_eq!(output.status.code(), Some(1));
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("blank.txt") && stderr.contains("non-whitespace"),
        "{stderr}"
    );
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_custom_system_prompt_is_used() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("system.txt");
    std::fs::write(&path, "You are a pirate.").unwrap();
    let (url, requests) = serve(vec![text_reply("Arr.")]).await;
    let output = run(
        args(&[
            "--base-url",
            &url,
            "--system",
            path.to_str().unwrap(),
            "--once",
            "hi",
        ]),
        "",
    )
    .await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(
        requests.lock().unwrap()[0]["messages"][0]["content"],
        "You are a pirate."
    );
}

#[tokio::test]
async fn a_bad_tool_root_is_an_error() {
    let output = run(args(&["--tool-root", "/no/such/dir", "--once", "hi"]), "").await;
    assert_eq!(output.status.code(), Some(1));
    assert!(
        text(&output.stderr).contains("--tool-root"),
        "{}",
        text(&output.stderr)
    );
}

#[tokio::test]
async fn usage_errors_exit_with_status_2() {
    for bad in [&["--max-steps", "0"][..], &["--bogus"][..]] {
        let output = run(args(bad), "").await;
        assert_eq!(output.status.code(), Some(2), "{bad:?}");
    }
}

#[tokio::test]
async fn environment_variables_configure_the_backend() {
    let (url, requests) = serve(vec![text_reply("ok")]).await;
    let output = tokio::task::spawn_blocking(move || {
        Command::new(BIN)
            .args(["--once", "hi"])
            .env("DJINN_AGENT_BASE_URL", url)
            .env("DJINN_AGENT_MODEL", "from-env")
            .stdin(Stdio::null())
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(requests.lock().unwrap()[0]["model"], "from-env");
}

#[tokio::test]
async fn the_interactive_session_answers_runs_commands_and_exits_on_eof() {
    let (url, _) = serve(vec![text_reply("Hello there.")]).await;
    let output = run(
        args(&["--base-url", &url]),
        "/tools\nhi\n/exit\nignored after exit\n",
    )
    .await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    assert!(stdout.contains("calculator: "), "{stdout}");
    assert!(stdout.contains("Hello there.\n"), "{stdout}");
    assert!(!stdout.contains("ignored"), "{stdout}");
}

#[tokio::test]
async fn the_interactive_session_starts_even_when_the_server_is_down() {
    let url = dead_url();
    let output = run(args(&["--base-url", &url]), "/help\n").await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("warning") && stderr.contains(&url),
        "{stderr}"
    );
    assert!(text(&output.stdout).contains("/exit"));
}

#[tokio::test]
async fn a_message_to_a_dead_server_is_reported_and_the_session_continues() {
    let url = dead_url();
    let output = run(args(&["--base-url", &url]), "hello\n/help\n").await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(
        text(&output.stderr).contains("could not connect"),
        "{}",
        text(&output.stderr)
    );
    assert!(
        text(&output.stdout).contains("/reset"),
        "the session carried on to /help"
    );
}

// ---- a real Ctrl-C ----------------------------------------------------------------------------

/// Serves one hanging reply (some text, then nothing), then normal replies.
async fn serve_hanging_first() -> (String, Requests) {
    let requests = Requests::default();
    let seen = requests.clone();
    async fn chat(
        State((requests, count)): State<(Requests, Arc<Mutex<usize>>)>,
        body: Bytes,
    ) -> Response {
        requests
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&body).unwrap());
        let nth = {
            let mut count = count.lock().unwrap();
            *count += 1;
            *count
        };
        if nth == 1 {
            let first = format!(
                "data: {}\n\n",
                json!({"choices": [{"index": 0, "delta": {"content": "partial "}, "finish_reason": null}]})
            );
            let stream = futures::stream::once(async move {
                Ok::<_, std::convert::Infallible>(Bytes::from(first))
            })
            .chain(futures::stream::pending());
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        } else {
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(text_reply("second answer")))
                .unwrap()
        }
    }
    use futures::StreamExt;
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(|| async { "ok" }))
        .with_state((seen, Arc::new(Mutex::new(0usize))));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}/v1"), requests)
}

#[tokio::test]
async fn ctrl_c_cancels_the_turn_and_the_next_question_is_answered() {
    use std::io::{BufRead, BufReader, Write};

    let (url, requests) = serve_hanging_first().await;
    let outcome = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(BIN)
            .args(["--base-url", &url])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        // Collect output on threads so nothing blocks on a full pipe.
        let (out_tx, out_rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            // Chunks, not lines: streamed text has no trailing newline until the answer ends.
            let mut stdout = stdout;
            let mut buffer = [0u8; 4096];
            while let Ok(read) = std::io::Read::read(&mut stdout, &mut buffer) {
                if read == 0 {
                    break;
                }
                let _ = out_tx.send(String::from_utf8_lossy(&buffer[..read]).into_owned());
            }
        });
        let errors = Arc::new(Mutex::new(String::new()));
        let errors_for_thread = errors.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                errors_for_thread
                    .lock()
                    .unwrap()
                    .push_str(&format!("{line}\n"));
            }
        });

        writeln!(stdin, "first question").unwrap();
        // Wait until the server has the request and has sent its partial text.
        let started = std::time::Instant::now();
        // The first chunk may be only the prompt, so read until the partial text arrives.
        let mut seen = String::new();
        while !seen.contains("partial") {
            let remaining = std::time::Duration::from_secs(10).saturating_sub(started.elapsed());
            match out_rx.recv_timeout(remaining) {
                Ok(chunk) => seen.push_str(&chunk),
                Err(_) => break,
            }
        }
        let partial_seen = seen.contains("partial");

        // A real SIGINT, delivered the way a terminal would.
        let signalled = Command::new("kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));

        writeln!(stdin, "second question").unwrap();
        let second = out_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .map(|chunk| chunk.trim().to_owned());
        let waited = started.elapsed();

        writeln!(stdin, "/exit").unwrap();
        drop(stdin);
        let status = child.wait().unwrap();
        let errors = errors.lock().unwrap().clone();
        (
            partial_seen,
            signalled.success(),
            second,
            waited,
            status.success(),
            errors,
        )
    })
    .await
    .unwrap();

    let (partial_seen, signalled, second, waited, exited_cleanly, errors) = outcome;
    assert!(
        partial_seen,
        "never saw the first, partial answer. stderr: {errors}"
    );
    assert!(signalled, "could not send SIGINT");
    assert_eq!(
        second.as_deref(),
        Ok("second answer"),
        "the second question was not answered after Ctrl-C (waited {waited:?}). stderr: {errors}"
    );
    assert!(exited_cleanly, "stderr: {errors}");
    assert!(errors.contains("turn cancelled"), "{errors}");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let second_messages = requests[1]["messages"].as_array().unwrap();
    let contents: Vec<_> = second_messages
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert!(
        !contents.contains(&"first question"),
        "the cancelled turn must not be in the history: {contents:?}"
    );
}

#[tokio::test]
async fn a_malformed_base_url_is_a_usage_error_and_no_request_is_sent() {
    let (url, requests) = serve(vec![text_reply("x")]).await;
    let output = run(args(&["--base-url", "localhost:8080", "--once", "hi"]), "").await;

    assert_eq!(
        output.status.code(),
        Some(2),
        "clap usage errors exit with 2"
    );
    let stderr = text(&output.stderr);
    assert!(stderr.contains("--base-url"), "{stderr}");
    assert!(text(&output.stdout).is_empty());
    assert!(requests.lock().unwrap().is_empty());
    drop(url);
}

#[tokio::test]
async fn an_empty_model_name_is_a_usage_error_naming_the_flag() {
    let output = run(args(&["--model", "", "--once", "hi"]), "").await;

    assert_eq!(output.status.code(), Some(2));
    assert!(
        text(&output.stderr).contains("--model"),
        "{}",
        text(&output.stderr)
    );
}

#[tokio::test]
async fn the_api_key_never_appears_in_any_output() {
    const KEY: &str = "sk-very-secret-key-123";
    // A failing request, a successful one, and a rejected key: none may print the key.
    let (url, _) = serve(vec![text_reply("fine")]).await;
    let outputs = [
        run(
            args(&["--base-url", &dead_url(), "--api-key", KEY, "--once", "hi"]),
            "",
        )
        .await,
        run(
            args(&["--base-url", &url, "--api-key", KEY, "--once", "hi"]),
            "",
        )
        .await,
        run(args(&["--api-key", "bad key", "--once", "hi"]), "").await,
    ];
    for output in outputs {
        for stream in [&output.stdout, &output.stderr] {
            let shown = text(stream);
            assert!(!shown.contains(KEY), "{shown}");
            assert!(!shown.contains("bad key"), "{shown}");
        }
    }
}

#[tokio::test]
async fn plain_and_approval_are_accepted_with_piped_input_and_tools_run_without_asking() {
    let (url, requests) = serve(vec![
        tool_reply("calculator", r#"{"expression":"2+2"}"#),
        text_reply("It is 4."),
    ])
    .await;
    let output = run(
        args(&["--base-url", &url, "--plain", "--approval", "auto"]),
        "2+2?\n",
    )
    .await;

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), "It is 4.\n");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn an_invalid_approval_mode_exits_2_naming_the_flag_and_the_values() {
    let output = run(args(&["--approval", "sometimes"]), "").await;
    assert_eq!(output.status.code(), Some(2));
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("--approval") && stderr.contains("ask") && stderr.contains("auto"),
        "{stderr}"
    );
}

#[tokio::test]
async fn piped_output_never_contains_terminal_control_sequences() {
    let (url, _) = serve(vec![text_reply("plain words")]).await;
    let output = run(args(&["--base-url", &url]), "hello\n").await;
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(!output.stdout.contains(&0x1b), "{:?}", text(&output.stdout));
    assert!(!output.stderr.contains(&0x1b), "{:?}", text(&output.stderr));
}
