import numpy as np
from mypkgs.processor.array_process import broadcast_to_any
from mypkgs.reader.ncreader import NCreader

def read2file(pressurefile, singlefile, varlist):
    """
    pressurefile: ERA5 files(filename or path) with pressure levels data
    singlefile: ERA5 files(filename or path) with single levels data
    varlist: variables to read (for pressurefile)
    """
    NCR = NCreader(pressurefile)
    NCR.auto_read(varlist=varlist)
    db = NCR.databox
    NCR.close()
    NCR = NCreader(singlefile)
    NCR.auto_read(read_all=True)
    db2 = NCR.databox
    NCR.close()
    db.merge(db2)
    p_4d = db["pressure_level"].data * 100 # Pa to hPa
    p_4d = broadcast_to_any(p_4d, db["u"][...].shape, n=1)
    mslp_4d = np.broadcast_to(db["sp"].data, (db["pressure_level"].data.shape[0], *db["msl"].data.shape))
    mslp_4d = np.transpose(mslp_4d, axes=(1,0,2,3))
    mask = mslp_4d < p_4d
    for key, variable in db.field.items():
        if len(variable.data.shape) == 4:
            db.field[key].data[mask] = np.nan
    return db