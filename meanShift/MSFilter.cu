__global__ filtering( Point3D *points, int width, int height, int hs, int hr, Point3D *result ){ 

    int tidx = blockDim.x * blockIdx.x + threadIdx.x ;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y ;

    if ( tidx < width && tidy < height ){ 

        int left = (tidx - hs ) < 0 ? 0 : (tidx - hs ) ;
        int right = (tidx + hs ) >= width ? width : (tidx + hs ) ;
        int top = ( tidy - hs ) < 0 ? 0 : ( tidy - hs ) ;
        int bottom = ( tidy + hs ) >= height ? height : ( tidy + hs ) ; 

        // Point5D .

        Point5D PtCur ;
        Point5D PtPrev ;
        Point5D PtSum ;
        Point5D Pt ;
        
        int step = 0 ;

        do { 

            PtPrev.MSPoint5DCopy( PtCur ) ;
            PtSum.MSPoint5DSet( 0 , 0 , 0 , 0 , 0 ) ;
            NumPts = 0 ;
            
            for( int hy = top ; hy < bottom ; hy ++ ){ 
                for( int hx = left ; hx < right ; hx ++ ){ 

                    Pt.MSPoint5DSet( hx , hy , points[ hx + hy * width ].l , points[ hx + hy * width ].g , points[ hx + hy * width ].b ) ;
                    Pt.PointLab() ;

                    if( Pt.MSPoint5DColorDistance(PtCur) < hr ){ 
                        PtSum.MSPointAcccm( Pt ) ;
                        NumPts ++ ;
                     }

                 }
             }

             PtSum.MSPoint5DScale( 1.0 / Numpts ) ;
             PtCur.MSPoint5DCopy( PtSum );
             step ++ ;

        }while ( (PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS) ) ;
 
        PtCur.PointRGB() ;

        result[ tidx + tidy * width ].l = PtCur.l ;
        result[ tidx + tidy * width ].g = PtCur.g ;
        result[ tidx + tidy * width ].b = PtCur.b ;

     }


 }



MSFiltering_d(Point3D *points, int width, int height, int hs, int hr, Point3D *output) {
    
    dim3 ThreadNum( 32 , 32 ) ;

    dim3 BlockNum( 1 ,  1 , 1 ) ;

    BlockNum.x = width / ThreadNum.x + 1 ;
    BlockNum.y = height / ThreadNum.y + 1 ;
    
    filtering<<< BlockNum , ThreadNum >>>( points , width , height , hs , hr , output ) ;
  }