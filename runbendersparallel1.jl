load("bendersserial")



function solveBendersParallel(nscen::Integer)

    scenarioData = monteCarloSample(probdata,1:nscen)

    stage1sol = solveExtensive(probdata,1)
    
    clpmaster = ClpModel()
    np = nprocs()

    ncol1 = probdata.firstStageData.ncol
    nrow1 = probdata.firstStageData.nrow
    nrow2 = probdata.secondStageTemplate.nrow
    # add \theta variables for cuts
    thetaidx = [(ncol1+1):(ncol1+nscen)]
    clp_load_problem(clpmaster, probdata.Amat, probdata.firstStageData.collb,
        probdata.firstStageData.colub, probdata.firstStageData.obj, 
        probdata.firstStageData.rowlb, probdata.firstStageData.rowub)
    zeromat = SparseMatrixCSC(int32(nrow1),int32(nscen),ones(Int32,nscen+1),Int32[],Float64[])
    clp_add_columns(clpmaster, -1e8*ones(nscen), Inf*ones(nscen),
        (1/nscen)*ones(nscen), zeromat)

    @everywhere clp_load_problem(clpsubproblem, probdata.Wmat, probdata.secondStageTemplate.collb,
        probdata.secondStageTemplate.colub, probdata.secondStageTemplate.obj,
        probdata.secondStageTemplate.rowlb, probdata.secondStageTemplate.rowub)

    thetasol = -1e8*ones(nscen)

    converged = false
    niter = 0
    mastertime = 0.
    while true
        Tx = d.Tmat*stage1sol
        # solve benders subproblems
        nviolated = 0
        #print("current solution is [")
        #for i in 1:ncol1
        #    print("$(stage1sol[i]),")
        #end
        #println("]")
        scen_count = 1
        next_scen() = (idx=scen_count; scen_count+=1; idx)
        new_violated() = (nviolated += 1)
        @sync for p in 1:np
            if p != myid() || np == 1
                @spawnlocal while true
                    s = next_scen()
                    if s > nscen
                        break
                    end
                    optval, subgrad = remote_call_fetch(p,solveSubproblem,scenarioData[s][1]-Tx,scenarioData[s][2]-Tx)
                    #println("For scen $s, optval is $optval and model value is $(thetasol[s])")
                    if (optval > thetasol[s] + 1e-7)
                        new_violated()
                        #print("adding cut: [")
                        # add (0-based) cut to master
                        cutvec = Float64[]
                        cutcolidx = Int32[]
                        for k in 1:ncol1
                         #   print("$(subgrad[k]),")
                            if abs(subgrad[k]) > 1e-10
                                push(cutvec,-subgrad[k])
                                push(cutcolidx,k-1)
                            end
                        end
                        #println("]")
                        push(cutvec,1.)
                        push(cutcolidx,ncol1+s-1)
                        cutnnz = length(cutvec)
                        cutlb = optval-dot(subgrad,stage1sol)

                        clp_add_rows(clpmaster, 1, [cutlb], [1e25], Int32[0,cutnnz], cutcolidx, cutvec)
                    end
                end
            end
        end

        if nviolated == 0
            break
        end
        println("Generated $nviolated violated cuts")
        # resolve master
        t = time()
        clp_initial_solve(clpmaster)
        mastertime += time() - t
        @assert clp_is_proven_optimal(clpmaster)
        sol = clp_get_col_solution(clpmaster)
        stage1sol = sol[1:ncol1]
        thetasol = sol[(ncol1+1):end]
        niter += 1
    end

    println("Optimal objective is: $(clp_get_obj_value(clpmaster)), $niter iterations")
    println("Time in master: $mastertime sec")

end

s = ARGS[1]
nscen = int(ARGS[2])
d = SMPSData(strcat(s,".cor"),strcat(s,".tim"),strcat(s,".sto"))
for p in 1:nprocs()
    remote_call_fetch(p,setGlobalProbData,d)
end
@time solveBendersParallel(nscen)
