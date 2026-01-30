use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use super::DataProcessor;

pub struct StandardScaler;

impl DataProcessor for StandardScaler 
{
    fn get_name(&self) -> &str 
    {
        "Standard Scaler"
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> 
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        for j in 0..cols 
        {
            let col = data.get_col(j);
            let mean = col.iter().sum::<f64>() / rows as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rows as f64;
            let std = var.sqrt();

            if std > 0.0 
            {
                for i in 0..rows 
                {
                    let val = (data.get(i, j) - mean) / std;
                    result.set(i, j, val);
                }
            }
        }
        result
    }
}