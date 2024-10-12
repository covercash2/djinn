const BYTE_UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

#[extend::ext]
pub impl u64 {
    fn to_kib(&self) -> String {
        let value = self.div_ceil(1024);
        format!("{value}KiB")
    }

    fn to_mib(&self) -> String {
        let value = self.div_ceil(1024 ^ 2);
        format!("{value}MiB")
    }

    fn to_gib(&self) -> String {
        let value = self.div_ceil(1024 ^ 3);
        format!("{value}GiB")
    }

    fn fit_to_bytesize(&self) -> String {
        let (value, order) = std::iter::repeat(())
            .enumerate()
            .map(|(i, ())| i)
            .scan(*self, |state, i| {
                if *state >= 1024 {
                    *state /= 1024;
                    Some((*state, i + 1))
                } else {
                    None
                }
            })
            .last()
            .unwrap_or((*self, 0));

        let unit = BYTE_UNITS[order];

        format!("{value}{unit}")
    }
}
