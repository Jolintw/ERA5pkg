import numpy as np
from mypkgs.processor.array_process import broadcast_to_any
from mypkgs.reader.ncreader import NCreader

def read2file(pressurefile, singlefile, varlist):
    """
    pressurefile: ERA5 files(filename or path) with pressure levels data\n
    singlefile: ERA5 files(filename or path) with single levels data\n
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
    
    sp_4d = np.broadcast_to(db["sp"].data, (db["pressure_level"].data.shape[0], *db["sp"].data.shape))
    sp_4d = np.transpose(sp_4d, axes=(1,0,2,3))
    mask = sp_4d < p_4d
    for key, variable in db.field.items():
        if len(variable.data.shape) == 4:
            db.field[key].data[mask] = np.nan
    db.add_field("p", p_4d, attr={"unit:":"Pa"})
    return db

def find_plevel(databox, p):
        datap = databox["pressure_level"].data
        level = np.argmin(np.abs(datap - p))
        if np.abs(datap[level] - p) > 1:
             print("p not match")
        return level