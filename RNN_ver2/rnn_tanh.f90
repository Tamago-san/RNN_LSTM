subroutine rnn_traning_own_fortran(in_node,out_node,rnn_node,traning_step,rnn_step,&
                    sample_num,epoch,epsi,g,&
                    u_trT,s_tr_dataT,u_rnnT,s_rnnT,w_outT,w_rnnT,w_inT,Tre_CH)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rnn_node,traning_step,rnn_step,sample_num,epoch
    real(8),    intent(inout) :: epsi,g
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
    real(8)     W_rnn(rnn_node,rnn_node)
    real(8)     W_in(rnn_node,in_node)
    real(8)     out_dEdw(out_node,rnn_node)
    real(8)     rnn_dEdw(rnn_node,rnn_node)
    real(8)     in_dEdw(rnn_node,in_node)
    
    real(8)     out_delta(traning_step+1,out_node)
    real(8)     rnn_delta(traning_step+1,rnn_node)
    
    real(8)     R_tr(0:traning_step+1,rnn_node)
    real(8)     R_rnn(0:rnn_step+1,rnn_node)
    real(8)     z_tr(traning_step,rnn_node)
    real(8)     z_rnn(rnn_step,rnn_node)
    real(8)     drdz,tmp,u_tmp(1:in_node),r_tmp(rnn_node)
    integer(4)  i,j ,k,iepo,isample,istep
    
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
    call inverse_matrix3(u_trT,u_tr,in_node,traning_step)
    call inverse_matrix3(s_tr_dataT,s_tr_data,out_node,traning_step)
    call inverse_matrix(u_rnnT,u_rnn,in_node,rnn_step)
    !０代入orsyokiti
    call random_number(W_out)
    call random_number(W_rnn)
    call random_number(W_in )
!    W_out=W_out/(rnn_node)**0.5
    W_rnn=W_rnn/(rnn_node)**0.5
    W_in=W_in/(rnn_node)**0.5
    R_tr =0.d0
    R_rnn=0.d0
    s_tr =0.d0
    s_rnn=0.d0
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !トレーニング
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    open(20,file='./data_out/tmp.csv')
    do i=1,traning_step
        write(20,*) i,u_tr(i,in_node,1:2),s_tr_data(i,out_node,1:2)
    enddo
    do iepo=1,epoch
        if(mod(iepo,10)==0) write(*,*) "epoch === " ,iepo
        do isample=1,sample_num
            !初期化
            out_dEdw=0.d0
            rnn_dEdw=0.d0
            in_dEdw=0.d0
            out_delta=0.d0
            rnn_delta=0.d0
            !順伝播計算
        

            call rnn_forward3(u_tr,s_tr,R_tr,z_tr,traning_step,rnn_node,isample)
            
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
                        tmp =tmp +(rnn_delta(istep+1,k) * W_rnn(k,i))
                    enddo
                    drdz= 4.d0 /(( exp(z_tr(istep,i))+exp(-z_tr(istep,i)) )**2)

                    rnn_delta(istep,i)=tmp*drdz*g
                enddo
                do i=1,rnn_node
                do j=1,rnn_node
                    rnn_dEdw(i,j) =rnn_dEdw(i,j)+ rnn_delta(istep,i) * R_tr(istep-1,j)
                enddo
                enddo

                !W_in
                do i=1,rnn_node
                do j=1,in_node
                    in_dEdw(i,j)=in_dEdw(i,j)+rnn_delta(istep,i) * u_tr(istep,j,isample)
                enddo
                enddo

            enddo
            !更新
            do i=1,in_node
            do j=1,rnn_node
                W_in(j,i)  = W_in(j,i) - epsi*in_dEdw(j,i)
            enddo
            enddo
            do i=1,rnn_node
            do j=1,rnn_node
                W_rnn(j,i) =W_rnn(j,i) - epsi*rnn_dEdw(j,i)
            enddo
            enddo
!            write(*,*) rnn_dEdw(5,1),W_rnn(5,1)
!            write(*,*) R_tr(50,:)
            do i=1,rnn_node
            do j=1,out_node
                W_out(j,i) =W_out(j,i) - epsi*(out_dEdw(j,i)+0.001d0*W_out(j,i) )
!                W_out(j,i) =W_out(j,i) -0.001 *(out_dEdw(j,i))
            enddo
            enddo
            
        enddo
        R_tr(0,:)=R_tr(traning_step,:)
        Tre_CH(1,iepo) = (sum(abs(out_dEdw)) )**2
        Tre_CH(2,iepo) = (sum(abs(rnn_dEdw)) )**2
        Tre_CH(3,iepo) = (sum(abs(in_dEdw)) )**2
!        write(*,*) out_dEdw(1,5),W_out(1,5)
    enddo
    Tre_CH(1,:) =Tre_CH(1,:)/Tre_CH(1,1)
    Tre_CH(2,:) =Tre_CH(2,:)/Tre_CH(2,1)
    Tre_CH(3,:) =Tre_CH(3,:)/Tre_CH(3,1)

    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !テスト
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    R_rnn(0,:)=R_tr(traning_step,:)
    call rnn_forward(u_rnn,s_rnn,R_rnn,z_rnn,rnn_step,rnn_node)
            
!    do i=1,rnn_step
!        write(20,*) i,s_rnn(i,1:out_node)
!    enddo


    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    !パイソンに出力
    !＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    call inverse_matrix(s_rnn,s_rnnT,rnn_step,out_node)
    call inverse_matrix(W_out,W_outT,out_node,rnn_node)
    call inverse_matrix(W_rnn,W_rnnT,rnn_node,rnn_node)
    call inverse_matrix(W_in ,W_inT ,rnn_node,in_node)
    
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
contains
    subroutine rnn_forward3(u_f,s_f,R_f,z_f,nstep,nnode,nsample)
        real(8) u_f(:,:,:),s_f(:,:),R_f(:,:),z_f(:,:)
        integer nstep,nnode
        real(8) wu(nnode), wr(nnode)
        integer f1,f2,f,fistep,nsample
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu(:)=0.d0
            wr(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu(f1) = wu(f1) + W_in(f1,f2) * u_f(fistep,f2,nsample)
                enddo
                do f2=1,rnn_node
                    wr(f1) = wr(f1) + W_rnn(f1,f2) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1)=wu(f1) + wr(f1)

                R_f(fistep,f1) = tanh(z_f(fistep,f1)*g)
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

    subroutine rnn_forward(u_f,s_f,R_f,z_f,nstep,nnode)
        real(8) u_f(:,:),s_f(:,:),R_f(:,:),z_f(:,:)
        integer nstep,nnode
        real(8) wu(nnode), wr(nnode)
        integer f1,f2,f,fistep
!        call rnn_function(R_f,u_f,z_f,nstep)
!!!!!!!!

        do fistep=1,nstep
            wu(:)=0.d0
            wr(:)=0.d0
            
            do f1=1,rnn_node
                do f2=1,in_node
                    wu(f1) = wu(f1) + W_in(f1,f2) * u_f(fistep,f2)
                enddo
                do f2=1,rnn_node
                    wr(f1) = wr(f1) + W_rnn(f1,f2) *R_f(fistep-1,f2)
                enddo
                z_f(fistep,f1)=wu(f1) + wr(f1)

                R_f(fistep,f1) = tanh(z_f(fistep,f1)*g)
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
    end subroutine rnn_forward

    subroutine inverse_matrix3(A_A,B_B,v2,v1)
        real(8) A_A(:,:),B_B(:,:,:)
        integer v1,v2,v11,v22,v33
        integer f1,f2,f3
        
        v11=size(A_A,1)
        v22=size(A_A,2)

        do f3=1,sample_num
        do f1=1,traning_step
        do f2=1,v2
            B_B(f1,f2,f3) = A_A(f2,f1+(f3-1)*traning_step)
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
end subroutine rnn_traning_own_fortran
