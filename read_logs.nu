def "read logs" [] {
	tail ~/.local/state/ollama_tui/tui.log | lines | each {|line| $line | from json}
}
