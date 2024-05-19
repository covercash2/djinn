def "djinn complete" [
	--protocol: string = "http"
	--addr: string = "localhost"
	--port: int = 8080
	--endpoint: string = "complete"
	--sample_len: int = 100,
	prompt: string,
] {
	let url = $"($protocol)://($addr):($port)/($endpoint)"
	let payload = {
		prompt: $prompt,
		sample_len: $sample_len,
	}
	http post --content-type application/json $url $payload
}

def "djinn run server" [
	--backend: string = "cuda"
	--config: string = "test"
	--debug
] {
	let features = if $backend == "cpu" {
		[]
	} else {
		["--features" $"djinn-core/($backend)"]
	}
	let build_mode = if $debug {
		[]
	} else {
		["--release"]
	}

	let cargo_args = $build_mode ++ $features
	let djinn_args = ["server" "--name" $config]

	run-external "cargo" "run" ...$cargo_args "--" ...$djinn_args
	cargo run --release --features djinn-core/cuda -- server --name test
}
