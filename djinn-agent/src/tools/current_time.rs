//! `current_time`: the current UTC date and time.

use std::time::{SystemTime, UNIX_EPOCH};

use djinn_core::values::ToolName;
use futures::future::BoxFuture;
use std::sync::LazyLock;

use serde_json::{Value, json};

use super::{Tool, ToolError};

/// Returns the current date and time in UTC.
pub struct CurrentTime;

impl Tool for CurrentTime {
    fn name(&self) -> &ToolName {
        static NAME: LazyLock<ToolName> = LazyLock::new(|| ToolName::from_static("current_time"));
        &NAME
    }

    fn description(&self) -> &str {
        "Get the current date and time in UTC, as an RFC 3339 timestamp such as 2026-10-26T14:30:00Z. Use it whenever the answer depends on today's date or the time."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }

    fn call(&self, _arguments: Value) -> BoxFuture<'_, Result<String, ToolError>> {
        Box::pin(async {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| ToolError::Failed("the system clock is before 1970".to_owned()))?;
            Ok(rfc3339_utc(now.as_secs() as i64))
        })
    }
}

/// Formats Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` (UTC).
///
/// Written out by hand instead of pulling in a date crate for one format. The calendar arithmetic
/// is Howard Hinnant's `civil_from_days` algorithm for the proleptic Gregorian calendar.
pub fn rfc3339_utc(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let seconds_of_day = unix_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds_of_day / 3600,
        seconds_of_day % 3600 / 60,
        seconds_of_day % 60
    )
}

/// Days since 1970-01-01 to (year, month, day).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_timestamps_format_correctly() {
        // Reference values computed independently (GNU `date -u -d @N`).
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(
            rfc3339_utc(951_782_400),
            "2000-02-29T00:00:00Z",
            "2000 is a leap year"
        );
        assert_eq!(rfc3339_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        assert_eq!(rfc3339_utc(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_709_251_200), "2024-03-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_735_689_599), "2024-12-31T23:59:59Z");
        assert_eq!(rfc3339_utc(1_735_689_600), "2025-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(2_147_483_647), "2038-01-19T03:14:07Z");
        assert_eq!(rfc3339_utc(4_102_444_800), "2100-01-01T00:00:00Z");
    }

    #[test]
    fn century_years_follow_the_gregorian_rules() {
        // 1900 and 2100 are not leap years; 2000 is.
        assert_eq!(rfc3339_utc(-2_203_891_200), "1900-03-01T00:00:00Z");
        assert_eq!(rfc3339_utc(4_107_542_400), "2100-03-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_868_800), "2000-03-01T00:00:00Z");
    }

    #[test]
    fn times_before_1970_work() {
        assert_eq!(rfc3339_utc(-1), "1969-12-31T23:59:59Z");
        assert_eq!(rfc3339_utc(-86_400), "1969-12-31T00:00:00Z");
    }

    #[test]
    fn consecutive_days_never_skip_or_repeat() {
        // Walk four years of days and check the date advances by exactly one day each time.
        let mut previous = rfc3339_utc(1_704_067_200 - 86_400); // 2023-12-31
        for day in 0..(366 + 365 + 365 + 365) {
            let current = rfc3339_utc(1_704_067_200 + day * 86_400);
            assert!(current > previous, "{previous} -> {current}");
            previous = current;
        }
        assert_eq!(previous, "2027-12-31T00:00:00Z");
    }

    /// Parses `YYYY-MM-DDTHH:MM:SSZ` back to Unix seconds, independently of `civil_from_days`.
    fn parse(timestamp: &str) -> i64 {
        let bytes = timestamp.as_bytes();
        assert_eq!(bytes.len(), 20, "{timestamp}");
        assert_eq!(
            (
                bytes[4], bytes[7], bytes[10], bytes[13], bytes[16], bytes[19]
            ),
            (b'-', b'-', b'T', b':', b':', b'Z')
        );
        let number = |range: std::ops::Range<usize>| timestamp[range].parse::<i64>().unwrap();
        let (year, month, day) = (number(0..4), number(5..7), number(8..10));
        let days_in_month = |y: i64, m: i64| match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
            _ => 28,
        };
        let mut days = 0;
        for y in 1970..year {
            days += if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                366
            } else {
                365
            };
        }
        for m in 1..month {
            days += days_in_month(year, m);
        }
        days += day - 1;
        days * 86_400 + number(11..13) * 3600 + number(14..16) * 60 + number(17..19)
    }

    #[tokio::test]
    async fn the_tool_returns_a_timestamp_close_to_the_system_clock() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let result = CurrentTime.call(json!({})).await.unwrap();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let parsed = parse(&result);
        assert!(
            parsed >= before - 1 && parsed <= after + 1,
            "{result} not within [{before}, {after}]"
        );
    }

    #[tokio::test]
    async fn arguments_are_ignored() {
        assert!(
            CurrentTime
                .call(json!({"timezone": "Mars/Base"}))
                .await
                .is_ok()
        );
        assert!(CurrentTime.call(Value::Null).await.is_ok());
    }

    #[test]
    fn declares_no_required_parameters() {
        let parameters = CurrentTime.parameters();
        assert_eq!(parameters["type"], "object");
        assert!(parameters.get("required").is_none());
    }
}
