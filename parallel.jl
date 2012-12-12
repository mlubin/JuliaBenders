load("parallelbase")

type CounterWrapper
    x
end

const nscen = 20

function asyncparalleltest()
    np = nprocs()
    d = MasterData()
    @assert np <= nscen

    # (data, scenario number)
    tasks = Array((SolType,Int),0)
    # initialize
    for i in 1:nscen
        push(tasks,(d.sol,i))
    end
    
    finishedDict = Dict{Real,Int}()
    finishedDict[d.sol] = 0
    nstarted = CounterWrapper(1)
    ndone = CounterWrapper(0)
    @everywhere srand(10)

    @sync for p in 1:np
        if p == myid()
            continue
        end
        @spawnlocal begin
            while d.status != "Done" #&& nstarted.x <= 5
                # send out new task
                t = shift(tasks)
                result = remote_call_fetch(p,subproblem,t)
                push(d.cuts,result)
                #println("Waiting for $p")
                #println("Proc $p is done, returned $(d.cuts[end])")
                nsolved = finishedDict[t[1]]
                if (nsolved+1)/nscen > .6 # if solved for 60% of scenarios, resolve master
                    masterproblem(d)
                    println("Solution of round $(d.niter) is $(d.sol)")
                    nstarted.x += 1
                    println("Started round $(nstarted.x)")
                    
                    for i in 1:nscen
                        push(tasks,(d.sol,i))
                    end
                    finishedDict[d.sol] = 0
                    # invert sign to mark semi-complete
                    nsolved = -nsolved
                end
                nsolved += sign(nsolved+0.5)
                finishedDict[t[1]] = nsolved
                if -nsolved == nscen
                    ndone.x += 1
                    println("Finished all scenarios, $(ndone.x) rounds complete")
                    # could remove from dictionary
                end
                
            end
        end
    end

    println("We're done. $(nstarted.x) rounds started, $(d.niter) calls to master, $(ndone.x) rounds completely finished")


end

@time asyncparalleltest()

@time begin
@everywhere srand(10)
for i in 1:5
    pmap(subproblem,ones(nscen))
end
end

@time begin
@everywhere srand(10)
for i in 1:5
    curproc = 1
    assignment = Dict{Int,Int}()
    for s in 1:nscen
        if curproc > nprocs()
            curproc = 1
        end
        if curproc == myid()
            curproc += 1
        end
        assignment[s] = curproc
        curproc += 1
    end
    #println(assignment)
    @sync for s in 1:nscen
        @spawnlocal remote_call_fetch(assignment[s],subproblem,1.)
    end
end
end
