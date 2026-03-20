pub mod i_model;  // IModel trait
pub mod linreg;
pub mod logreg;
pub mod tree;
pub mod knn;
pub mod random_forest;
pub mod svm;
pub mod gradient_boosting;
pub mod polynomial;

pub use knn::KnnWrapper;
pub use linreg::LinRegWrapper;
pub use logreg::LogRegWrapper;
pub use tree::TreeWrapper;
pub use random_forest::RandomForestWrapper;
pub use svm::SvmWrapper;
pub use polynomial::PolyRegWrapper;
pub use gradient_boosting::GradientBoostingWrapper;
pub use i_model::IModel;

pub mod factory;
