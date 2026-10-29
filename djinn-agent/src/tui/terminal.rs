//! The only file that touches the real terminal: raw mode, alternate screen, restore on drop and
//! panic (design D8). It is left out of the coverage numbers because it needs a real terminal.

use std::io::{self, Stdout, stdout};

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    supports_keyboard_enhancement,
};
use crossterm::{cursor, execute};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

/// Puts the terminal in TUI mode and puts it back when dropped.
pub struct TerminalGuard {
    keyboard_enhancement: bool,
}

impl TerminalGuard {
    /// Enters raw mode and the alternate screen, with mouse and paste reporting. A panic from here
    /// on restores the terminal before the message is printed.
    pub fn enter() -> io::Result<(Self, Terminal<CrosstermBackend<Stdout>>)> {
        install_panic_hook();
        enable_raw_mode()?;
        let mut out = stdout();
        // From here on, `Drop` undoes whatever succeeded.
        let mut guard = Self {
            keyboard_enhancement: false,
        };
        execute!(
            out,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;
        // Lets Shift-Enter be told apart from Enter, where the terminal can report it.
        if supports_keyboard_enhancement().unwrap_or(false) {
            execute!(
                out,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
            guard.keyboard_enhancement = true;
        }
        let terminal = Terminal::new(CrosstermBackend::new(out))?;
        Ok((guard, terminal))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore(self.keyboard_enhancement);
    }
}

/// Undoes everything [`TerminalGuard::enter`] did. Errors are ignored: there is nothing left to do
/// about them, and this also runs while panicking.
fn restore(keyboard_enhancement: bool) {
    let mut out = stdout();
    if keyboard_enhancement {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(
        out,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen,
        cursor::Show
    );
    let _ = disable_raw_mode();
}

/// Restores the terminal before the previous panic hook prints, so the message stays readable.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore(true);
        previous(info);
    }));
}
