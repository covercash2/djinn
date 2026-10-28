//! Parameters that steer sampling.

const fn valid_temperature(value: &f64) -> bool {
    value.is_finite() && *value >= 0.0 && *value <= 2.0
}

const fn valid_top_p(value: &f64) -> bool {
    value.is_finite() && *value > 0.0 && *value <= 1.0
}

const fn valid_max_tokens(value: &usize) -> bool {
    *value >= 1
}

number_value! {
    /// Sampling temperature: 0 (deterministic) to 2.
    Temperature(f64), "temperature", valid_temperature, "must be a number from 0 to 2",
    schema_type = Number,
    minimum = Some(0.0), maximum = Some(2.0), exclusive_minimum = None
}

number_value! {
    /// Nucleus sampling cutoff: above 0 and at most 1.
    TopP(f64), "top_p", valid_top_p, "must be a number above 0 and at most 1",
    schema_type = Number,
    minimum = None, maximum = Some(1.0), exclusive_minimum = Some(0.0)
}

number_value! {
    /// A limit on generated tokens: a whole number of at least 1.
    MaxTokens(usize), "max_tokens", valid_max_tokens, "must be a whole number of at least 1",
    schema_type = Integer,
    minimum = Some(1.0), maximum = None, exclusive_minimum = None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_boundaries() {
        for ok in [0.0, 0.7, 2.0, 1e-7] {
            assert!(Temperature::new(ok).is_ok(), "{ok}");
        }
        for bad in [-0.1, 2.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Temperature::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn top_p_boundaries() {
        for ok in [0.01, 0.9, 1.0] {
            assert!(TopP::new(ok).is_ok(), "{ok}");
        }
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(TopP::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn max_tokens_must_be_positive() {
        assert!(MaxTokens::new(1).is_ok());
        assert!(MaxTokens::new(0).is_err());
    }

    #[test]
    fn errors_name_the_parameter_and_the_value() {
        let error = Temperature::new(5.0).unwrap_err().to_string();
        assert!(
            error.contains("temperature") && error.contains('5'),
            "{error}"
        );
        assert!(MaxTokens::new(0)
            .unwrap_err()
            .to_string()
            .contains("max_tokens"));
        assert!(TopP::new(7.0).unwrap_err().to_string().contains("top_p"));
    }

    #[test]
    fn constants_are_checked_at_compile_time() {
        const DEFAULT: Temperature = Temperature::from_const(1e-7);
        const LIMIT: MaxTokens = MaxTokens::from_const(100);
        assert_eq!(DEFAULT.get(), 1e-7);
        assert_eq!(LIMIT.get(), 100);
    }

    #[test]
    #[should_panic(expected = "constant out of range")]
    fn an_invalid_constant_panics() {
        let _ = std::hint::black_box(Temperature::from_const(9.0));
    }

    #[test]
    fn deserialization_validates_and_serialization_is_the_bare_number() {
        assert!(serde_json::from_str::<Temperature>("3").is_err());
        assert!(serde_json::from_str::<TopP>("0").is_err());
        assert!(serde_json::from_str::<MaxTokens>("0").is_err());
        assert!(serde_json::from_str::<MaxTokens>("-1").is_err());
        let temperature: Temperature = serde_json::from_str("0.5").unwrap();
        assert_eq!(serde_json::to_string(&temperature).unwrap(), "0.5");
        let tokens: MaxTokens = serde_json::from_str("64").unwrap();
        assert_eq!(serde_json::to_string(&tokens).unwrap(), "64");
    }

    #[test]
    fn parses_from_text_for_command_lines() {
        assert_eq!("0.3".parse::<Temperature>().unwrap().get(), 0.3);
        assert!("hot".parse::<Temperature>().is_err());
        assert!("3".parse::<Temperature>().is_err());
        assert_eq!("12".parse::<MaxTokens>().unwrap().get(), 12);
    }

    #[test]
    fn schemas_carry_the_bounds() {
        let temperature = serde_json::to_value(schemars::schema_for!(Temperature)).unwrap();
        assert_eq!(temperature["minimum"], 0.0);
        assert_eq!(temperature["maximum"], 2.0);
        let top_p = serde_json::to_value(schemars::schema_for!(TopP)).unwrap();
        assert_eq!(top_p["exclusiveMinimum"], 0.0);
        let tokens = serde_json::to_value(schemars::schema_for!(MaxTokens)).unwrap();
        assert_eq!(tokens["type"], "integer");
        assert_eq!(tokens["minimum"], 1.0);
    }
}
