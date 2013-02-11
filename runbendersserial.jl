require("bendersserial")

s = ARGS[1]
nscen = int(ARGS[2])
d = SMPSData(string(s,".cor"),string(s,".tim"),string(s,".sto"))
@time solveBendersSerial(d,nscen)
