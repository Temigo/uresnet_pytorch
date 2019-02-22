from iotools_sparse import io_larcv_sparse
from iotools_dense import io_larcv_dense
# from sparse_dataloader import SparseDataLoader
from iotools_ppn import io_larcv_ppn


def io_factory(flags):
    if flags.IO_TYPE == 'larcv_sparse':  # SSCN I/O
        return io_larcv_sparse(flags)
    if flags.IO_TYPE == 'larcv_dense':  # Dense I/O
        return io_larcv_dense(flags)
    # if flags.IO_TYPE == 'sparse_dataloader':
    #     return SparseDataLoader(flags)
    if flags.IO_TYPE == 'larcv_sparse_ppn':
        return io_larcv_ppn(flags)
    raise NotImplementedError
