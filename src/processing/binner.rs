use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use super::DataProcessor;

pub struct Binner 
{
    bins: usize,
}

impl Binner 
{
    pub fn new(bins: usize) -> Self 
    {
        Self 
        { 
            bins 
        }
    }
}

impl DataProcessor for Binner 
{
    fn get_name(&self) -> &str 
    {
        "Equal-width Binner"
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> 
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        for j in 0..cols 
        {
            let col = data.get_col(j);
            let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;

            if range > 0.0 
            {
                for i in 0..rows 
                {
                    let val = data.get(i, j);
                    let mut bin = ((val - min) / range * self.bins as f64).floor();
                    
                    if bin >= self.bins as f64 
                    {
                        bin = (self.bins - 1) as f64;
                    }
                    
                    result.set(i, j, bin);
                }
            }
        }
        result
    }
}