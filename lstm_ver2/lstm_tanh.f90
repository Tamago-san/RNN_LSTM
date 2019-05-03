subroutine lstm_traning_own_fortran(in_node,out_node,rnn_node,traning_step,rnn_step,&
                    sample_num,epoch,epsi,g,&
                    u_trT,s_tr_dataT,u_rnnT,s_rnnT,w_outT,w_rnnT,w_inT,Tre_CH,TR_loss)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rnn_node,traning_step,rnn_step,sample_num,epoch
    real(8),    intent(inout) :: epsi,g
    real(8),    intent(inout) :: TR_loss( epoch*sample_num)
    real(8),    intent(inout) :: w_outT(rnn_node,out_node)
    real(8),    intent(inout) :: w_rnnT(rnn_node,rnn_node)
    real(8),    intent(inout) :: w_inT(in_node,rnn_node)
    real(8),    intent(inout) :: u_trT(in_node,traning_step) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) :: s_tr_dataT(out_node,traning_step)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) :: u_rnnT(in_node ,rnn_step) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) :: s_rnnT(out_node,rnn_step)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) :: Tre_CH(3,epoch)
    real(8)     u_tr(traning_step,in_node,sample_num) !今は一次元、列サイズはトレーニング時間
    real(8)     s_tr(traning_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8)     s_tr_data(traning_step,out_node,sample_num)  !出力次元数、列サイズはトレーニング時間
    real(8)     u_rnn(rnn_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8)     s_rnn(rnn_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8)     W_out(out_node,rnn_node)
    real(8)     W_rnn(rnn_node,rnn_node,4)
    real(8)     W_in(rnn_node,in_node,4)
    real(8)     W_cel(rnn_node,4)
    real(8)     out_dEdw(out_node,rnn_node)
    real(8)     rnn_dEdw(rnn_node,rnn_node,4)
    real(8)     in_dEdw(rnn_node,in_node,4)
    real(8)     cel_dEdw(rnn_node,4)
    real(8)     out_delta(traning_step+1,out_node)
    real(8)     cel_delta(traning_step+1,rnn_node)
    real(8)     rnn_delta(traning_step+1,rnn_node,4)
    real(8)     rnn_delta_tmp(traning_step+1,rnn_node)
    
    real(8)     R_tr(0:traning_step+1,rnn_node)
    real(8)     R_rnn(0:rnn_step+1,rnn_node)
    real(8)     C_tr(0:traning_step,rnn_node)
    real(8)     C_rnn(0:rnn_step,rnn_node)
    real(8)     Z_tr(traning_step+1,rnn_node,4)
    real(8)     Z_rnn(rnn_step+1,rnn_node,4)
    real(8)     drdz,tmp,u_tmp(1:in_node),r_tmp(rnn_node)
    real(8)     loss,iepsi,ep
    integer(4)  i,j ,k,iepo,isample,istep,inode
    
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    write(*,*) "in_node      ",in_node
    write(*,*) "out_node     ",out_node
    write(*,*) "rnn_node     ",rnn_node
    write(*,*) "traning_step",traning_step
    write(*,*) "rnn_step     ",rnn_step
    write(*,*) "EPOCH        ",epoch
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""

    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !初期化
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !転置
    call inverse_matrix3(u_trT,u_tr,in_node,traning_step,sample_num)
    call inverse_matrix3(s_tr_dataT,s_tr_data,out_node,traning_step,sample_num)
    call inverse_matrix(u_rnnT,u_rnn,in_node,rnn_step)
    !０代入orsyokiti
    call random_seed
    call random_number(W_out)
    call random_number(W_rnn)
    call random_number(W_in)
    call random_number(W_cel)

    W_out=1.d-2 *(2.d0*W_out-1.d0)
    W_rnn=1.d-2 *(2.d0*W_rnn-1.d0)
    W_in =1.d-2 *(2.d0*W_in -1.d0)
    W_cel=1.d-2 *(2.d0*W_cel-1.d0)
    R_tr =0.d0
    R_rnn=0.d0
    C_tr =0.d0
    C_rnn=0.d0
    s_tr =0.d0
    s_rnn=0.d0
    iepsi=epsi
    loss=0.d0
    
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !トレーニング
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    open(20,file='./data_out/tmp.csv')
    open(21,file='./data_out/loss.dat')
    do i=1,traning_step
        write(20,*) i,u_tr(i,in_node,1:2),s_tr_data(i,out_node,1:2)
    enddo
    do iepo=1,epoch
        if(mod(iepo,50)==0) write(*,*) "================================================"
        if(mod(iepo,50)==0) write(*,*) "   EOCH   === " ,iepo
        if(mod(iepo,50)==0) write(*,*) "------------------------------------------------"
        if(mod(iepo,50)==0) write(*,*) "loss       === " ,loss
        if(mod(iepo,50)==0) write(*,*) "iepsi      === " ,iepsi
        if(mod(iepo,50)==0) write(*,*) "------------------------------------------------"
        do isample=1,sample_num
            !初期化
            out_dEdw=0.d0
            rnn_dEdw=0.d0
             in_dEdw =0.d0
            cel_dEdw=0.d0
            
            out_delta =0.d0
            cel_delta =0.d0
            rnn_delta =0.d0
            rnn_delta_tmp=0.d0
            !順伝播計算
            call rnn_forward3(u_tr,s_tr,R_tr,C_tr,Z_tr,traning_step,rnn_node,isample)
            loss=0.d0
            do istep=1,traning_step
            do inode=1,out_node
                loss=loss+abs( s_tr(istep,inode) - s_tr_data(istep,inode,isample) )
            enddo
            enddo
            loss=loss/dble(traning_step*out_node)
            TR_loss((iepo-1)*sample_num+isample )=loss
            !逆伝播計算
            do istep=traning_step,1,-1
                !W_out
                out_delta(istep,:)= (s_tr(istep,:) - s_tr_data(istep,:,isample))
                
                do i=1,out_node
                do j=1,rnn_node
                    out_dEdw(i,j) = out_dEdw(i,j) + out_delta(istep,i) * R_tr(istep,j)
                enddo
                enddo

                !RNN
                do i=1,rnn_node
                    tmp=0.d0
                    drdz=0.d0
                    do k=1,out_node
                        tmp =tmp +(out_delta(istep,k) * W_out(k,i))
                    enddo
                    do k=1,rnn_node
                        tmp =tmp +(rnn_delta(istep+1,k,3) * W_rnn(k,i,3))
                    enddo
                    rnn_delta(istep,i,4)  = d_sigmoid(Z_tr(istep,i,4)) *tanh(C_tr(istep,i) )*tmp
                    rnn_delta_tmp(istep,i)= sigmoid(Z_tr(istep,i,4)) *d_tanh(C_tr(istep,i) )*tmp
                enddo
                do i=1,rnn_node
                    cel_delta(istep,i) =rnn_delta_tmp(istep,i)&
                                        +sigmoid(Z_tr(istep+1,i,1))*cel_delta(istep+1,i)&
                                        +W_cel(i,1)*rnn_delta(istep+1,i,1)&
                                        +W_cel(i,2)*rnn_delta(istep+1,i,2)&
                                        +W_cel(i,4)*rnn_delta(istep  ,i,4)
                enddo
                do i=1,rnn_node
                    rnn_delta(istep,i,3) =   sigmoid(Z_tr(istep,i,2))*d_tanh(Z_tr(istep,i,3))*cel_delta(istep,i)
                    rnn_delta(istep,i,2) = d_sigmoid(Z_tr(istep,i,2))*  tanh(Z_tr(istep,i,3))*cel_delta(istep,i)
                    rnn_delta(istep,i,1) = d_sigmoid(Z_tr(istep,i,1))*  C_tr(istep-1,i)      *cel_delta(istep,i)
                enddo
                do k=1,4
                do i=1,rnn_node
                do j=1,rnn_node
                    rnn_dEdw(i,j,k) =rnn_dEdw(i,j,k)+ rnn_delta(istep,i,k) * R_tr(istep-1,j)
                enddo
                enddo
                enddo
                
                
                do i=1,rnn_node
                    cel_dEdw(i,1) =cel_dEdw(i,1)+ rnn_delta(istep,i,1) * C_tr(istep-1,i)
                    cel_dEdw(i,2) =cel_dEdw(i,2)+ rnn_delta(istep,i,2) * C_tr(istep-1,i)
                    cel_dEdw(i,4) =cel_dEdw(i,4)+ rnn_delta(istep,i,4) * C_tr(istep,i)
                enddo
                

                !W_in
                do k=1,4
                do i=1,rnn_node
                do j=1,in_node
                    in_dEdw(i,j,k)=in_dEdw(i,j,k)+rnn_delta(istep,i,k) * u_tr(istep,j,isample)
                enddo
                enddo
                enddo

            enddo
            !更新
            do k=1,4
            do i=1,in_node
            do j=1,rnn_node
                W_in(j,i,k)  = W_in(j,i,k) - iepsi*in_dEdw(j,i,k)
            enddo
            enddo
            enddo
            do k=1,4
            do i=1,rnn_node
            do j=1,rnn_node
                W_rnn(j,i,k) =W_rnn(j,i,k) - iepsi*rnn_dEdw(j,i,k)
            enddo
            enddo
            enddo
            
            do i=1,rnn_node
                W_cel(i,1) =W_cel(i,1) - iepsi*cel_dEdw(i,1)
                W_cel(i,2) =W_cel(i,2) - iepsi*cel_dEdw(i,2)
                W_cel(i,4) =W_cel(i,4) - iepsi*cel_dEdw(i,4)
            enddo
!            write(*,*) rnn_dEdw(5,1),W_rnn(5,1)
!            write(*,*) R_tr(50,:)
            do i=1,rnn_node
            do j=1,out_node
!                W_out(j,i) =W_out(j,i) - iepsi*(out_dEdw(j,i)+0.001d0*W_out(j,i) )
                W_out(j,i) =W_out(j,i) -iepsi *(out_dEdw(j,i))
            enddo
            enddo
            
        enddo
        R_tr(0,:)=R_tr(traning_step,:)
        C_tr(0,:)=C_tr(traning_step,:)
        Tre_CH(1,iepo) = (sum(abs(out_dEdw)) )**2
        Tre_CH(2,iepo) = (sum(abs(rnn_dEdw)) )**2
        Tre_CH(3,iepo) = (sum(abs(in_dEdw)) )**2
        write(21,*) loss
!!        write(*,*) out_dEdw(1,5),W_out(1,5)
    enddo
    Tre_CH(1,:) =log(Tre_CH(1,:)/Tre_CH(1,1))
    Tre_CH(2,:) =log(Tre_CH(2,:)/Tre_CH(2,1))
    Tre_CH(3,:) =log(Tre_CH(3,:)/Tre_CH(3,1))

    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !テスト
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    R_rnn(0,:)=R_tr(traning_step,:)
    C_rnn(0,:)=C_tr(traning_step,:)
    call rnn_forward2(u_rnn,s_rnn,R_rnn,c_rnn,z_rnn,rnn_step,rnn_node)
            
!    do i=1,rnn_step
!        write(20,*) i,s_rnn(i,1:out_node)
!    enddo


    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !パイソンに出力
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    call inverse_matrix(s_rnn,s_rnnT,rnn_step,out_node)
    call inverse_matrix(W_out,W_outT,out_node,rnn_node)
!    call inverse_matrix(W_rnn3,W_rnnT,rnn_node,rnn_node)
!    call inverse_matrix(W_in3 ,W_inT ,rnn_node,in_node)
    
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    EXIT     Fortran90 !    "
    write(*,*) "-------------------------------"
    write(*,*) "in_node     ",in_node
    write(*,*) "out_node    ",out_node
    write(*,*) "rnn_node     ",rnn_node
    write(*,*) "traning_step",traning_step
    write(*,*) "rnn_step     ",rnn_step
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
    close(20)
    close(21)
contains
    subroutine rnn_forward3(u_f,s_f,R_f,C_f,Z_f,nstep,nnode,nsample)
        real(8) u_f(:,:,:),s_f(:,:),R_f(0:traning_step+1,rnn_node),C_f(0:traning_step,rnn_node),z
        integer nstep,nnode
        real(8) wu1(nnode),wu2(nnode),wu3(nnode),wu4(nnode)
        real(8) wr1(nnode),wr2(nnode),wr3(nnode),wr4(nnode)
        real(8) Z_f(nstep,rnn_node,4)
        integer f1,f2,f,fistep,nsample
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu1(:)=0.d0
            wu2(:)=0.d0
            wu2(:)=0.d0
            wu4(:)=0.d0
            wr1(:)=0.d0
            wr2(:)=0.d0
            wr2(:)=0.d0
            wr4(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu1(f1) = wu1(f1) + W_in(f1,f2,1) * u_f(fistep,f2,nsample)
                    wu2(f1) = wu2(f1) + W_in(f1,f2,2) * u_f(fistep,f2,nsample)
                    wu3(f1) = wu3(f1) + W_in(f1,f2,3) * u_f(fistep,f2,nsample)
                    wu4(f1) = wu4(f1) + W_in(f1,f2,4) * u_f(fistep,f2,nsample)
                enddo
                do f2=1,rnn_node
                    wr1(f1) = wr1(f1) + W_rnn(f1,f2,1) *R_f(fistep-1,f2)
                    wr2(f1) = wr2(f1) + W_rnn(f1,f2,2) *R_f(fistep-1,f2)
                    wr3(f1) = wr3(f1) + W_rnn(f1,f2,3) *R_f(fistep-1,f2)
                    wr4(f1) = wr4(f1) + W_rnn(f1,f2,4) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1,1)=wu1(f1) + wr1(f1) + W_cel(f1,1)*c_f(fistep-1,f1)
                z_f(fistep,f1,2)=wu2(f1) + wr2(f1) + W_cel(f1,2)*c_f(fistep-1,f1)
                z_f(fistep,f1,3)=wu3(f1) + wr3(f1)
                
                C_f(fistep,f1)=sigmoid( z_f(fistep,f1,1) ) * C_f(fistep-1,f1)&
                                + sigmoid(z_f(fistep,f1,2)) * tanh(z_f(fistep,f1,3))
                
                Z_f(fistep,f1,4)=wu4(f1) + wr4(f1) + W_cel(f1,4)*C_f(fistep,f1)

                R_f(fistep,f1) = tanh(C_f(fistep,f1))*sigmoid(z_f(fistep,f1,4))
            enddo

            do f1=1,out_node
                s_f(fistep,f1)=0.d0
                do f2=1,rnn_node
                    s_f(fistep,f1) = s_f(fistep,f1)+ W_out(f1,f2)*R_f(fistep,f2)
                enddo
            enddo
!        write(*,*) s_f(fistep,:)
        enddo
!!!!!!!!!!
    end subroutine rnn_forward3
    subroutine rnn_forward2(u_f,s_f,R_f,C_f,Z_f,nstep,nnode)
        real(8) u_f(:,:),s_f(:,:),R_f(0:rnn_step+1,rnn_node),C_f(0:rnn_step,rnn_node)
        integer nstep,nnode
        real(8) wu1(nnode),wu2(nnode),wu3(nnode),wu4(nnode)
        real(8) wr1(nnode),wr2(nnode),wr3(nnode),wr4(nnode)
        real(8) Z_f(nstep,rnn_node,4)
        integer f1,f2,f,fistep
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu1(:)=0.d0
            wu2(:)=0.d0
            wu2(:)=0.d0
            wu4(:)=0.d0
            wr1(:)=0.d0
            wr2(:)=0.d0
            wr2(:)=0.d0
            wr4(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu1(f1) = wu1(f1) + W_in(f1,f2,1) * u_f(fistep,f2)
                    wu2(f1) = wu2(f1) + W_in(f1,f2,2) * u_f(fistep,f2)
                    wu3(f1) = wu3(f1) + W_in(f1,f2,3) * u_f(fistep,f2)
                    wu4(f1) = wu4(f1) + W_in(f1,f2,4) * u_f(fistep,f2)
                enddo
                do f2=1,rnn_node
                    wr1(f1) = wr1(f1) + W_rnn(f1,f2,1) *R_f(fistep-1,f2)
                    wr2(f1) = wr2(f1) + W_rnn(f1,f2,2) *R_f(fistep-1,f2)
                    wr3(f1) = wr3(f1) + W_rnn(f1,f2,3) *R_f(fistep-1,f2)
                    wr4(f1) = wr4(f1) + W_rnn(f1,f2,4) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1,1)=wu1(f1) + wr1(f1) + W_cel(f1,1)*c_f(fistep-1,f1)
                z_f(fistep,f1,2)=wu2(f1) + wr2(f1) + W_cel(f1,2)*c_f(fistep-1,f1)
                z_f(fistep,f1,3)=wu3(f1) + wr3(f1)
                
                C_f(fistep,f1)=sigmoid( z_f(fistep,f1,1) ) * C_f(fistep-1,f1)&
                                + sigmoid(z_f(fistep,f1,2)) * tanh(z_f(fistep,f1,3))
                
                Z_f(fistep,f1,4)=wu4(f1) + wr4(f1) + W_cel(f1,4)*C_f(fistep,f1)

                R_f(fistep,f1) = tanh(C_f(fistep,f1))*sigmoid(z_f(fistep,f1,4))
            enddo

            do f1=1,out_node
                s_f(fistep,f1)=0.d0
                do f2=1,rnn_node
                    s_f(fistep,f1) = s_f(fistep,f1)+ W_out(f1,f2)*R_f(fistep,f2)
                enddo
            enddo
!        write(*,*) s_f(fistep,:)
        enddo
!!!!!!!!!!
    end subroutine rnn_forward2
    
    function sigmoid(x)
        real(8) x, sigmoid   ! 順番は関係ありません
        sigmoid = 1.d0 / (1.d0 + exp(-x) )
    end function sigmoid
    
    function d_sigmoid(x)
        real(8) x, d_sigmoid   ! 順番は関係ありません
        d_sigmoid = sigmoid(x)*(1-sigmoid(x))
    end function d_sigmoid
    
    function d_tanh(x)
        real(8) x, d_tanh   ! 順番は関係ありません
        d_tanh = 4.d0 /(( exp(x)+exp(-x) )**2)
    end function d_tanh
    
    subroutine inverse_matrix3(A_A,B_B,v2,v1,nsample)
        real(8) A_A(:,:),B_B(:,:,:)
        integer v1,v2,v11,v22,v33,nsample
        integer f1,f2,f3
        
        v11=size(A_A,1)
        v22=size(A_A,2)

        do f3=1,nsample
        do f1=1,v1
        do f2=1,v2
            B_B(f1,f2,f3) = A_A(f2,f1+(f3-1)*v1)
        enddo
        enddo
        enddo
    end subroutine inverse_matrix3
    subroutine inverse_matrix(A_A,B_B,v2,v1)
        real(8) A_A(:,:),B_B(:,:)
        integer v1,v2,v11,v22,v33
        integer f1,f2,f3
        
        v11=size(A_A,1)
        v22=size(A_A,2)


        do f1=1,v1
        do f2=1,v2
            B_B(f1,f2) = A_A(f2,f1)
        enddo
        enddo

    end subroutine inverse_matrix
end subroutine lstm_traning_own_fortran
