//! `calculator`: arithmetic and nothing else.
//!
//! A small recursive-descent parser over decimal numbers, `+ - * /`, parentheses and unary minus.
//! There are no variables, functions, exponent operators or implicit multiplication, so the model
//! cannot make the tool do anything but arithmetic.

use djinn_core::values::ToolName;
use futures::future::BoxFuture;
use std::sync::LazyLock;

use serde_json::{json, Value};

use super::{string_arg, Tool, ToolError};

/// Longest expression accepted, in characters.
const MAX_LENGTH: usize = 200;
/// Deepest nesting of parentheses and unary minus accepted.
const MAX_DEPTH: usize = 50;

/// Evaluates an arithmetic expression.
pub struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &ToolName {
        static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("calculator"));
        &NAME
    }

    fn description(&self) -> &str {
        "Evaluate an arithmetic expression and return the number. Supports decimal numbers, + - * /, parentheses and unary minus. Use it for any calculation instead of working it out yourself."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate, for example \"(17 + 3) * 2.5\""
                }
            },
            "required": ["expression"]
        })
    }

    fn call(&self, arguments: Value) -> BoxFuture<'_, Result<String, ToolError>> {
        Box::pin(async move {
            let expression = string_arg(&arguments, "expression")?;
            evaluate(expression)
                .map(format_number)
                .map_err(ToolError::Failed)
        })
    }
}

/// Evaluates `expression`, or explains why it cannot.
pub(crate) fn evaluate(expression: &str) -> Result<f64, String> {
    let chars: Vec<char> = expression.chars().collect();
    if chars.len() > MAX_LENGTH {
        return Err(format!(
            "expression is too long ({} characters, the limit is {MAX_LENGTH})",
            chars.len()
        ));
    }
    if chars.iter().all(|c| c.is_whitespace()) {
        return Err("empty expression".to_owned());
    }

    let mut parser = Parser { chars, position: 0 };
    let value = parser.expression(0)?;
    parser.skip_whitespace();
    if let Some(&c) = parser.chars.get(parser.position) {
        return Err(unsupported(c, parser.position));
    }
    if !value.is_finite() {
        return Err("the result is not a finite number".to_owned());
    }
    Ok(value)
}

fn unsupported(c: char, position: usize) -> String {
    format!("unsupported expression: unexpected `{c}` at position {position}")
}

struct Parser {
    chars: Vec<char>,
    position: usize,
}

impl Parser {
    fn skip_whitespace(&mut self) {
        while self
            .chars
            .get(self.position)
            .is_some_and(|c| c.is_whitespace())
        {
            self.position += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_whitespace();
        self.chars.get(self.position).copied()
    }

    /// expr = term (('+' | '-') term)*
    fn expression(&mut self, depth: usize) -> Result<f64, String> {
        let mut value = self.term(depth)?;
        while let Some(op @ ('+' | '-')) = self.peek() {
            self.position += 1;
            let rhs = self.term(depth)?;
            value = if op == '+' { value + rhs } else { value - rhs };
        }
        Ok(value)
    }

    /// term = unary (('*' | '/') unary)*
    fn term(&mut self, depth: usize) -> Result<f64, String> {
        let mut value = self.unary(depth)?;
        while let Some(op @ ('*' | '/')) = self.peek() {
            self.position += 1;
            let rhs = self.unary(depth)?;
            if op == '/' {
                if rhs == 0.0 {
                    return Err("division by zero".to_owned());
                }
                value /= rhs;
            } else {
                value *= rhs;
            }
        }
        Ok(value)
    }

    /// unary = '-' unary | primary
    fn unary(&mut self, depth: usize) -> Result<f64, String> {
        if depth > MAX_DEPTH {
            return Err(format!(
                "expression is nested too deeply (limit {MAX_DEPTH})"
            ));
        }
        if self.peek() == Some('-') {
            self.position += 1;
            return Ok(-self.unary(depth + 1)?);
        }
        self.primary(depth)
    }

    /// primary = number | '(' expr ')'
    fn primary(&mut self, depth: usize) -> Result<f64, String> {
        match self.peek() {
            Some('(') => {
                self.position += 1;
                let value = self.expression(depth + 1)?;
                match self.peek() {
                    Some(')') => {
                        self.position += 1;
                        Ok(value)
                    }
                    Some(c) => Err(unsupported(c, self.position)),
                    None => Err("unsupported expression: missing closing `)`".to_owned()),
                }
            }
            Some(c) if c.is_ascii_digit() || c == '.' => self.number(),
            Some(c) => Err(unsupported(c, self.position)),
            None => Err("unsupported expression: it ends where a number was expected".to_owned()),
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        let start = self.position;
        let mut seen_dot = false;
        let mut digits = 0;
        while let Some(&c) = self.chars.get(self.position) {
            if c.is_ascii_digit() {
                digits += 1;
            } else if c == '.' && !seen_dot {
                seen_dot = true;
            } else {
                break;
            }
            self.position += 1;
        }
        if digits == 0 {
            return Err(unsupported('.', start));
        }
        let text: String = self.chars[start..self.position].iter().collect();
        text.parse::<f64>()
            .map_err(|_| format!("`{text}` is not a valid number"))
    }
}

/// Integers print without a fractional part; other values keep up to 12 decimals, so `0.1 + 0.2`
/// reads as `0.3`.
fn format_number(value: f64) -> String {
    let text = format!("{value:.12}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    match text {
        "-0" | "" => "0".to_owned(),
        other => other.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(expression: &str) -> Result<String, ToolError> {
        Calculator.call(json!({ "expression": expression })).await
    }

    #[tokio::test]
    async fn precedence_and_parentheses() {
        assert_eq!(run("2 + 3 * (4 - 1)").await.unwrap(), "11");
        assert_eq!(run("(2 + 3) * 4").await.unwrap(), "20");
        assert_eq!(
            run("10 - 4 - 3").await.unwrap(),
            "3",
            "subtraction is left associative"
        );
        assert_eq!(
            run("100 / 10 / 5").await.unwrap(),
            "2",
            "division is left associative"
        );
    }

    #[tokio::test]
    async fn unary_minus() {
        assert_eq!(run("-5 + 2").await.unwrap(), "-3");
        assert_eq!(run("--5").await.unwrap(), "5");
        assert_eq!(run("3 * -2").await.unwrap(), "-6");
        assert_eq!(run("-(2 + 3)").await.unwrap(), "-5");
    }

    #[tokio::test]
    async fn decimals_and_integer_formatting() {
        assert_eq!(run("1.5 * 2").await.unwrap(), "3", "integers have no `.0`");
        assert_eq!(run("0.1 + 0.2").await.unwrap(), "0.3");
        assert_eq!(run(".5 + .25").await.unwrap(), "0.75");
        assert_eq!(run("7 / 2").await.unwrap(), "3.5");
        assert_eq!(run("1 / 3").await.unwrap(), "0.333333333333");
        assert_eq!(
            run("0 * -1").await.unwrap(),
            "0",
            "negative zero reads as 0"
        );
    }

    #[tokio::test]
    async fn large_values() {
        assert_eq!(run("17 * 23").await.unwrap(), "391");
        assert_eq!(run("123456789 * 1000").await.unwrap(), "123456789000");
    }

    #[tokio::test]
    async fn division_by_zero_is_an_error() {
        let error = run("1 / 0").await.unwrap_err();
        assert!(error.to_string().contains("division by zero"), "{error}");
        assert!(run("1 / (2 - 2)").await.is_err());
    }

    #[tokio::test]
    async fn anything_but_arithmetic_is_rejected_without_evaluating() {
        for bad in [
            "system(\"ls\")",
            "2 ** 3",
            "2^3",
            "1e3",
            "x + 1",
            "2 (3)",
            "3 3",
            "1 2",
            "1..2",
            "$(whoami)",
            "`ls`",
            "1;2",
        ] {
            let error = run(bad).await.unwrap_err();
            assert!(
                error.to_string().contains("unsupported")
                    || error.to_string().contains("not a valid"),
                "`{bad}` gave: {error}"
            );
        }
    }

    #[tokio::test]
    async fn unbalanced_parentheses() {
        assert!(run("(1 + 2").await.is_err());
        assert!(run("1 + 2)").await.is_err());
        assert!(run("()").await.is_err());
    }

    #[tokio::test]
    async fn empty_and_incomplete_expressions() {
        assert!(run("").await.unwrap_err().to_string().contains("empty"));
        assert!(run("   ").await.unwrap_err().to_string().contains("empty"));
        assert!(run("1 +").await.is_err());
        assert!(run("*").await.is_err());
    }

    #[tokio::test]
    async fn overlong_and_deeply_nested_expressions_are_rejected() {
        let long = format!("1{}", " + 1".repeat(100));
        assert!(run(&long)
            .await
            .unwrap_err()
            .to_string()
            .contains("too long"));

        let deep = format!("{}1{}", "(".repeat(99), ")".repeat(99)); // 199 chars: under the length limit
        assert!(run(&deep)
            .await
            .unwrap_err()
            .to_string()
            .contains("nested too deeply"));

        let unary = format!("{}1", "-".repeat(150));
        assert!(run(&unary)
            .await
            .unwrap_err()
            .to_string()
            .contains("nested too deeply"));

        let ok = format!("{}1{}", "(".repeat(40), ")".repeat(40));
        assert_eq!(run(&ok).await.unwrap(), "1");
    }

    #[tokio::test]
    async fn the_longest_allowed_numbers_stay_finite() {
        // The length limit keeps every result far below `f64::MAX`, so the finiteness check in
        // `evaluate` is purely defensive.
        let nines = "9".repeat(190);
        let value = evaluate(&nines).unwrap();
        assert!(value.is_finite() && value > 1e189);
    }

    #[tokio::test]
    async fn missing_or_mistyped_argument() {
        let error = Calculator.call(json!({})).await.unwrap_err();
        assert!(
            matches!(&error, ToolError::InvalidArguments(m) if m.contains("expression")),
            "{error}"
        );
        let error = Calculator.call(json!({"expression": 5})).await.unwrap_err();
        assert!(
            matches!(&error, ToolError::InvalidArguments(m) if m.contains("string")),
            "{error}"
        );
    }

    #[test]
    fn declares_a_required_expression_parameter() {
        let parameters = Calculator.parameters();
        assert_eq!(parameters["required"], json!(["expression"]));
        assert_eq!(parameters["properties"]["expression"]["type"], "string");
    }
}
