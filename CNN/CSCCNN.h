class CSC_Convolutional_Neural_Network
{
  public:
    Input_Layer INPUT;
    CSC_Layer C_1;
    Convolutional_Layer C_3;
    Max_Pooling_Layer MP_4;
    Fully_Connected_Layer FC_7;
    Fully_Connected_Layer FC_8;
    Output_Layer FC_9;

    void init()
    {
	INPUT.init(1, 28, 28);
	C_1.init_1(&INPUT, 12, 2, 2);
	C_3.init_1(&C_1, 24, 4, 4, false, 0, 1.0);
	C_1.init_2(&C_3);
	MP_4.init_1(&C_3, 2, 2);
	C_3.init_2(&MP_4);
	FC_7.init_1(&MP_4, 256, true, 1);
	MP_4.init_2(&FC_7, true);
	FC_8.init_1(&FC_7, 96, false, 1);
	FC_7.init_2(&FC_8, false);
	FC_9.init_1(&FC_8, 10);
	FC_8.init_2(&FC_9, true);
	C_3.udout();
    }

    void calculate()
    {
#ifdef _DEBUG_Y_
	std::cout << "======\nY\n======\nC_1" << std::endl;
#endif
	C_1.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "C_3" << std::endl;
#endif
	C_3.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "C_5" << std::endl;
#endif
// C_5.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "END_Y" << std::endl;
#endif
    }

    void train(float *y)
    {
	FC_7.dout();
	FC_8.dout();
	calculate();
#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.calculate_delta(y);
#ifdef _DEBUG_T_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.calculate_delta(NULL);
#ifdef _DEBUG_T_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.calculate_delta(NULL);
#ifdef _DEBUG_T_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "C_5" << std::endl;
#endif
// C_5.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "C_3" << std::endl;
#endif
	C_3.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "==========================================" << std::endl;
#endif

#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "C_5" << std::endl;
#endif
// C_5.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "C_3" << std::endl;
#endif
	C_3.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "==========================================" << std::endl;
#endif

	FC_7.udout();
	FC_8.udout();
    }

    void change_weight(float eta)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "C_5" << std::endl;
#endif
// C_5.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "C_3" << std::endl;
#endif
	C_3.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.change_weight(eta);
    }

    void change_weight(CSC_Convolutional_Neural_Network *source, float eta)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.change_weight(&(source->FC_9), eta);
#ifdef _DEBUG_T_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.change_weight(&(source->FC_8), eta);
#ifdef _DEBUG_T_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.change_weight(&(source->FC_7), eta);
#ifdef _DEBUG_T_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.change_weight(&(source->MP_6),eta);
#ifdef _DEBUG_T_
	std::cout << "C_5" << std::endl;
#endif
// C_5.change_weight(&(source->C_5),eta);
#ifdef _DEBUG_T_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.change_weight(&(source->MP_4), eta);
#ifdef _DEBUG_T_
	std::cout << "C_3" << std::endl;
#endif
	C_3.change_weight(&(source->C_3), eta);
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.change_weight(&(source->C_1), eta);
    }

    void copy_weight(CSC_Convolutional_Neural_Network *source)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_9.copy_weight(&(source->FC_9));
#ifdef _DEBUG_T_
	std::cout << "FC_8" << std::endl;
#endif
	FC_8.copy_weight(&(source->FC_8));
#ifdef _DEBUG_T_
	std::cout << "FC_7" << std::endl;
#endif
	FC_7.copy_weight(&(source->FC_7));
#ifdef _DEBUG_T_
	std::cout << "MP_6" << std::endl;
#endif
// MP_6.copy_weight(&(source->MP_6));
#ifdef _DEBUG_T_
	std::cout << "C_5" << std::endl;
#endif
// C_5.copy_weight(&(source->C_5));
#ifdef _DEBUG_T_
	std::cout << "MP_4" << std::endl;
#endif
	MP_4.copy_weight(&(source->MP_4));
#ifdef _DEBUG_T_
	std::cout << "C_3" << std::endl;
#endif
	C_3.copy_weight(&(source->C_3));
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.copy_weight(&(source->C_1));
    }
};
