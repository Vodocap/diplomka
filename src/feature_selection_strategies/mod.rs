pub mod feature_selector;  // FeatureSelector trait
pub mod variance_selector;
pub mod correlation_selector;
pub mod chi_square_selector;
pub mod information_gain_selector;
pub mod mutual_information_selector;
pub mod smc_selector;
pub mod factory;

pub use variance_selector::VarianceSelector;
pub use correlation_selector::CorrelationSelector;
pub use chi_square_selector::ChiSquareSelector;
pub use information_gain_selector::InformationGainSelector;
pub use mutual_information_selector::MutualInformationSelector;
pub use smc_selector::SmcSelector;
pub use feature_selector::FeatureSelector;
