# Parse the ollama TUI log (JSON lines) into a table.
export def "read logs" [] {
	tail ~/.local/state/ollama_tui/tui.log | lines | each {|line| $line | from json}
}
