function A = prepare_senatedata()
%   A0: A numeric matrix (one row per student, one column per field)
%   fieldnames: A cell array of strings containing the extracted column headers

    T = readtable('114_congress.csv', 'Delimiter', ',');
    
    T = T(:, 4:end);

    A = table2array(T);
    A(A == 0) = -1;
    A(A == 0.5) = 0;

    fprintf("Senate data has %i bills and %i senators.\n", height(A), width(A))
end