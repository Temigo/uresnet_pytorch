from iotools_sparse import io_larcv_sparse
from iotools_dense import io_larcv_dense


def io_factory(flags):
    if flags.IO_TYPE == 'larcv_sparse':  # SSCN I/O
        return io_larcv_sparse(flags)
    if flags.IO_TYPE == 'larcv_dense':  # Dense I/O
        return io_larcv_dense(flags)
    raise NotImplementedError
    