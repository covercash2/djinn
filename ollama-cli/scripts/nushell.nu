
module ollama {
	const OLLAMA_HOST = "http://hoss:11434"

	export def "ollama get" [
		endpoint: string
		--host: string = $OLLAMA_HOST
	] {
		http get $"($host)/api/($endpoint)"
	}

	export def "ollama post" [
		endpoint: string
		payload: record
		--host: string = $OLLAMA_HOST
	] {
			http post --content-type application/json $"($host)/api/($endpoint)" $payload
	}

	export def "ollama list" [
		--host: string = $OLLAMA_HOST
	] {
		ollama get tags
	}

	export def "ollama download modelinfo" [
		names: list
		output_dir: path
		--host: string = $OLLAMA_HOST
	] {
		$names | each { |name|
			print $name
			ollama post show { name: $name }
			| save --force ($output_dir | path join $"($name).model.json")
		}
	}

	export def "ollama download modelinfo all" [
		output_dir: path
		--host: string = $OLLAMA_HOST
	] {
		let names = ollama list
		| get models.name
		ollama download modelfiles $names $output_dir --host $host
	}

	export def "ollama download modelfile all" [
		output_dir: path
		--host: string = $OLLAMA_HOST
	] {
		ollama list
		| get models.name
		| each {|name|
			ollama post show { name: $name }
			| get modelfile
			| save --force ($output_dir | path join $"($name).Modelfile")
		}
	}

	export def "ollama pull" [
		name: string
		--host: string = $OLLAMA_HOST
	] {
		ollama post pull { name: $name stream: false }
	}

}
