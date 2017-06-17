class Small_Convolutional_Neural_Network
{
  public:
    Input_Layer INPUT;
    Convolutional_Layer C_1;
    Max_Pooling_Layer MP_2;
    Fully_Connected_Layer FC_3;

    void init()
    {
	INPUT.init(1, 13, 13);
	C_1.init_1(&INPUT, 6, 5, 5, true, 0, 1.0);
	MP_2.init_1(&C_1, 3, 3);
	C_1.init_2(&MP_2);
	FC_3.init_1(&MP_2, 2, true, 1);
	MP_2.init_2(&FC_3, true);
	FC_3.init_2(NULL, false);
    }

    void calculate()
    {
#ifdef _DEBUG_Y_
	std::cout << "======\nY\n======\nC_1" << std::endl;
#endif
	C_1.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "FC_3" << std::endl;
#endif
	FC_3.calculate_y();
#ifdef _DEBUG_Y_
	std::cout << "END_Y" << std::endl;
#endif
    }

    void train(float *y)
    {
	calculate();
#ifdef _DEBUG_T_
	std::cout << "FC_3" << std::endl;
#endif
	FC_3.calculate_delta(y);
#ifdef _DEBUG_T_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.calculate_delta();
#ifdef _DEBUG_T_
	std::cout << "==========================================" << std::endl;
#endif

#ifdef _DEBUG_T_
	std::cout << "FC_3" << std::endl;
#endif
	FC_3.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.calculate_d_w();
#ifdef _DEBUG_T_
	std::cout << "==========================================" << std::endl;
#endif
    }

    void change_weight(float eta)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_9" << std::endl;
#endif
	FC_3.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.change_weight(eta);
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.change_weight(eta);
    }

    void change_weight(Small_Convolutional_Neural_Network *source, float eta)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_3" << std::endl;
#endif
	FC_3.change_weight(&(source->FC_3), eta);
#ifdef _DEBUG_T_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.change_weight(&(source->MP_2), eta);
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.change_weight(&(source->C_1), eta);
    }

    void copy_weight(Small_Convolutional_Neural_Network *source)
    {
#ifdef _DEBUG_T_
	std::cout << "FC_3" << std::endl;
#endif
	FC_3.copy_weight(&(source->FC_3));
#ifdef _DEBUG_T_
	std::cout << "MP_2" << std::endl;
#endif
	MP_2.copy_weight(&(source->MP_2));
#ifdef _DEBUG_T_
	std::cout << "C_1" << std::endl;
#endif
	C_1.copy_weight(&(source->C_1));
    }
};
