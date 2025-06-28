import ParallelStencil: INDICES
ix, iy, iz = INDICES[1], INDICES[2], INDICES[3]

function NNp(n)
    n%L+1
end

function NNp2(n)
    (n+1)%L+1
end

function NNm(n)
    (n+L-2)%L+1
end

function NNm2(n)
    (n+L-3)%L+1
end

macro d_xc(A) esc(:( ($A[$NNp($ix),$iy,$iz] - $A[$NNm($ix),$iy,$iz])/2 )) end
macro d_yc(A) esc(:( ($A[$ix,$NNp($iy),$iz] - $A[$ix,$NNm($iy),$iz])/2 )) end
macro d_zc(A) esc(:( ($A[$ix,$iy,$NNp($iz)] - $A[$ix,$iy,$NNm($iz)])/2 )) end

macro prd_d_xc(A,B) esc(:( ($A[$NNp($ix),$iy,$iz] * $B[$NNp($ix),$iy,$iz]
                         - $A[$NNm($ix),$iy,$iz] * $B[$NNm($ix),$iy,$iz])/2 )) end
macro prd_d_yc(A,B) esc(:( ($A[$ix,$NNp($iy),$iz] * $B[$ix,$NNp($iy),$iz]
                         - $A[$ix,$NNm($iy),$iz] * $B[$ix,$NNm($iy),$iz])/2 )) end
macro prd_d_zc(A,B) esc(:( ($A[$ix,$iy,$NNp($iz)] * $B[$ix,$iy,$NNp($iz)]
                         - $A[$ix,$iy,$NNm($iz)] * $B[$ix,$iy,$NNm($iz)])/2 )) end

macro d2_xyz(A)
    esc(:(
         ($A[$NNp2($ix),$iy,$iz] + $A[$NNm2($ix),$iy,$iz]
        + $A[$ix,$NNp2($iy),$iz] + $A[$ix,$NNm2($iy),$iz]
        + $A[$ix,$iy,$NNp2($iz)] + $A[$ix,$iy,$NNm2($iz)]
        - 6*$A[$ix,$iy,$iz]) * 0.25
    ))
end

function view_tuple(u)
    if size(u, 4) == 3
        return (@view(u[:,:,:,1]),@view(u[:,:,:,2]),@view(u[:,:,:,3]))
    end

    (@view(u[:,:,:,1]),@view(u[:,:,:,2]),@view(u[:,:,:,3]),@view(u[:,:,:,4]))
end
