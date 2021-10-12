using Statistics


function parse_folder(folder_dir)
    file_names = readdir(folder_dir)
    final_train_lls = Vector{Float64}()
    final_valid_lls = Vector{Float64}()
    final_test_lls = Vector{Float64}()
    for file_name in file_names
        if file_name[1] == '.'
            continue
        end
        dataset_name = split(file_name, '.')[1]
        raw_results = open(f -> read(f, String), "./$(folder_dir)/$(file_name)")
        final_results = split(raw_results, '\n')[end-1]
        final_train_ll = parse(Float64, split(final_results, ' ')[1])
        final_valid_ll = parse(Float64, split(final_results, ' ')[2])
        final_test_ll = parse(Float64, split(final_results, ' ')[3])
        # println("$(dataset_name) - ($(final_train_ll), $(final_valid_ll), $(final_test_ll))")

        train_lls = Vector{Float64}()
        valid_lls = Vector{Float64}()
        test_lls = Vector{Float64}()
        raw_data = split(raw_results, '\n')[1:end-1]
        for line in raw_data
            train_ll = parse(Float64, split(final_results, ' ')[1])
            valid_ll = parse(Float64, split(final_results, ' ')[2])
            test_ll = parse(Float64, split(final_results, ' ')[3])
            push!(train_lls, train_ll)
            push!(valid_lls, valid_ll)
            push!(test_lls, test_ll)
        end
        idx = argmax(valid_lls)
        println("$(dataset_name) - ($(train_lls[idx]), $(valid_lls[idx]), $(test_lls[idx]))")
        push!(final_train_lls, train_lls[idx])
        push!(final_valid_lls, valid_lls[idx])
        push!(final_test_lls, test_lls[idx])
    end
    final_train_lls, final_valid_lls, final_test_lls
end

main_folder_dir = ARGS[1]
_, _, test1 = parse_folder(main_folder_dir * "_1")
_, _, test2 = parse_folder(main_folder_dir * "_2")
_, _, test3 = parse_folder(main_folder_dir * "_3")
_, _, test4 = parse_folder(main_folder_dir * "_4")
_, _, test5 = parse_folder(main_folder_dir * "_5")

tests = hcat(test1, test2, test3, test4, test5)
print(mean(tests; dims = 2))
print(std(tests; dims = 2))