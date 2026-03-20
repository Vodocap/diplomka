use crate::embedded::{EmbeddedFeatureSelector, RandomForestSelector, RidgeSelector};

/// Factory pre embedded feature selection.
/// Pre klasifikaciu vytvori RandomForestSelector (korelacne proxy),
/// pre regresu RidgeSelector (L2 regularizacia, absolutne vahy koeficientov).
pub struct EmbeddedSelectorFactory;

impl EmbeddedSelectorFactory
{
    /// Vytvorí embedded selector podľa typu úlohy
    pub fn create_for_task(is_classification: bool) -> Box<dyn EmbeddedFeatureSelector>
    {
        if is_classification
        {
            Box::new(RandomForestSelector::new(true))
        }
        else
        {
            // Pre regresiu použijeme Ridge (L2 regularization)
            Box::new(RidgeSelector::new())
        }
    }
}
