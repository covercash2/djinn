export const BACKENDS = ["cuda" "cpu" "mac"]

def base_url [protocol: string, addr: string, port: int]: nothing -> string {
	$"($protocol)://($addr):($port)"
}

export def "djinn health" [
	--protocol: string = "http"
	--addr: string = "localhost"
	--port: int = 8080
] {
	http get $"(base_url $protocol $addr $port)/health-check"
}

export def "djinn complete" [
	--protocol: string = "http"
	--addr: string = "localhost"
	--port: int = 8080
	--extra: record = {}
	prompt: string
] {
	let url = $"(base_url $protocol $addr $port)/complete"
	let payload = {prompt: $prompt} | merge $extra
	http post --content-type application/json $url $payload
}

# Stream tokens from the server via SSE, printing each token as it arrives.
export def "djinn stream" [
	--protocol: string = "http"
	--addr: string = "localhost"
	--port: int = 8080
	prompt: string
] {
	let url = $"(base_url $protocol $addr $port)/complete/stream"
	let payload = {prompt: $prompt} | to json
	^curl -sN -X POST $url -H "Content-Type: application/json" -d $payload
	| lines
	| where { $in | str starts-with "data: " }
	| each { str replace "data: " "" | print --no-newline }
	ignore
}

# Compute CLIP similarity between a text prompt and an image file.
export def "djinn clip" [
	--protocol: string = "http"
	--addr: string = "localhost"
	--port: int = 8080
	prompt: string
	image: path       # path to image file
] {
	let url = $"(base_url $protocol $addr $port)/clip"
	let image_b64 = open --raw $image | encode base64
	let payload = {prompt: $prompt, image: $image_b64}
	http post --content-type application/json $url $payload
}

export def "djinn run server" [
	--backend: string = "mac"
	--config: string = "test"
	--debug
] {
	let features = if $backend == "cpu" {
		[]
	} else {
		["--features" $"djinn-core/($backend)"]
	}
	let build_mode = if $debug { [] } else { ["--release"] }

	run-external "cargo" "run" ...($build_mode ++ $features) "--" "server-config" "--name" $config
}
