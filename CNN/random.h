float r()
{
    return (rand() / (RAND_MAX + 1.0));
}

float R(float num)
{
    if ((rand() % 2) == 0)
    {
        return (0.1 * num * r());
    }
    else
    {
        return (-0.1 * num * r());
    }
}
