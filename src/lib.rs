use num::PrimInt;

pub enum GraphMatrixError {
    InvalidIndex,
    BoundsError,
}

/// Given identically-sized vectors representing row/column data, return a sparse matrix
/// representation.
fn compress<T: PrimInt>(row: Vec<T>, col: Vec<T>, n: usize) -> Result<(Vec<usize>, Vec<T>), GraphMatrixError> {

    let mut w: Vec<usize> = vec![0; n];
    let mut ja: Vec<T> = vec![T::zero(); col.len()];

    for v in &row {
        w[v.to_usize().ok_or(GraphMatrixError::InvalidIndex)?] += 1;
    }
    let ia = w.iter().fold(vec![0], |mut acc, val| {
        acc.push(val + acc.last().unwrap());
        acc
    });
    let mut w = ia.clone();
    if let Some(last) = w.last_mut() {
        *last = 0;
    }
    for (j, v) in col.into_iter().enumerate() {
        let rj = row[j].to_usize().ok_or(GraphMatrixError::InvalidIndex)?;
        let p = w[rj];
        ja[p] = v;
        w[rj] += 1;
    }

    Ok((ia, ja))
}

/// A GraphMatrix is a compressed sparse row matrix with no "value" vector. An element is 
/// said to exist when the col/row exists.
#[derive(Debug)]
pub struct GraphMatrix<T> {
    indptr: Vec<usize>,
    indices: Vec<T>,
}

impl<T> GraphMatrix<T> where T: PrimInt {

    pub fn dims(&self) -> (usize, usize) {
        (self.indptr.len() - 1, self.indptr.len() - 1)
    }

    pub fn ne(&self) -> usize {
        self.indices.len()
    }

    pub fn row(&self, r: T) -> Result<&[T], GraphMatrixError> {
        let ru = r.to_usize().ok_or(GraphMatrixError::InvalidIndex)?;
        if ru > self.indptr.len() - 2 {
            return Err(GraphMatrixError::BoundsError)
        }
        let start_index = unsafe { self.indptr.get_unchecked(ru) };
        let end_index = unsafe { self.indptr.get_unchecked(ru+1) };
        Ok(&self.indices[*start_index..*end_index])
    }

    pub fn row_len(&self, r: T) -> Result<usize, GraphMatrixError> {
        Ok(self.row(r)?.len())
    }

    pub fn has_index(&self, r: T, c: T) -> Result<bool, GraphMatrixError>
    {
        let row = self.row(r)?;
        let tc = T::from(c).ok_or(GraphMatrixError::InvalidIndex)?;
        Ok(row.binary_search(&tc).is_ok())
    }

    pub fn from_edgelist(edgelist: Vec<(T, T)>) -> Result<Self, GraphMatrixError> {
        let mut sorted_edgelist = edgelist;
        sorted_edgelist.sort_unstable();
        sorted_edgelist.dedup();
        let (ss, ds): (Vec<_>, Vec<_>) = sorted_edgelist.into_iter().unzip();

        let m1 = ss.last().ok_or(GraphMatrixError::InvalidIndex)?;
        let m2 = ds.iter().max().ok_or(GraphMatrixError::InvalidIndex)?;
        let m = m1
            .max(m2)
            .to_usize()
            .ok_or(GraphMatrixError::InvalidIndex)? + 1;
        let (indptr, indices) = compress(ss, ds, m)?;
        Ok(GraphMatrix {indptr, indices})
    }
}

#[derive(Debug)]
pub struct GraphMatrixIterator<'a, T: 'a> {
    gm: &'a GraphMatrix<T>,
    curr_rownum: T,
    curr_colptr: usize,
}

impl<'a, T: num::PrimInt> GraphMatrixIterator<'a, T> {
    pub fn new(g: &'a GraphMatrix<T>) -> Self {
        GraphMatrixIterator{gm: g, curr_rownum: T::zero(), curr_colptr: 0}
    }
}

impl<'a, T:num::PrimInt + std::fmt::Display> Iterator for GraphMatrixIterator<'a, T> where T: PrimInt {
    type Item = (T, T);
    fn next(&mut self) -> Option<(T, T)> {
        let row_data = self.gm.row(self.curr_rownum).ok()?;
        let v = (self.curr_rownum, row_data[self.curr_colptr]);
        self.curr_colptr += 1;
        if self.curr_colptr >= row_data.len() {
            self.curr_rownum = self.curr_rownum + T::one();
            self.curr_colptr = 0;
        }
        Some(v)
    }
}
