#ifndef NADARAYA_WATSON_NADARAYA_WATSON_H
#define NADARAYA_WATSON_NADARAYA_WATSON_H

#include <random>
#include <algorithm>
#include <iostream>
#include <nanoflann.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace nadaraya_watson {

    template<typename T, typename IT = size_t>
    class LimitedRadiusResultSet {

    public:
        const T radius;
        std::vector<IT> &indices;
        std::vector<T> &dists;
        IT max_size;
        IT count;
        std::mt19937 rng;

        inline LimitedRadiusResultSet(T radius_, std::vector<IT> &indices, std::vector<T> &dists, IT max_size) :
                radius(radius_), indices(indices), dists(dists), max_size(max_size), count(0) {
            init();
        }

        inline void init() { clear(); }

        inline void clear() {
            indices.clear();
            dists.clear();
            count = 0;
        }

        inline size_t size() const { return std::min(count, max_size); }

        inline bool full() const { return count >= max_size; }

        /**
         * Called during search to add an element matching the criteria.
         * @return true if the search should be continued, false if the results are
         * sufficient
         */
        inline bool addPoint(T dist, IT index) {
            if (count < max_size) {
                indices.push_back(index);
                dists.push_back(dist);
            } else {
                std::uniform_int_distribution<std::mt19937::result_type> uni_dist(0, count);
                IT idx = uni_dist(rng);
                if (idx < max_size) {
                    indices[idx] = index;
                    dists[idx] = dist;
                }
            }

            count += 1;
            return true;
        }

        inline T worstDist() const { return radius; }

    };


    template<typename T>
    class BufferAdaptor {
    public:
        explicit BufferAdaptor(const pybind11::buffer &data) : source(data.request()), ref(data.inc_ref()) {
            if (source.format != pybind11::format_descriptor<T>::format()) {
                throw std::runtime_error("Expected x-data type " + pybind11::format_descriptor<T>::format() +
                                         "but got " + source.format);
            }

            if (source.ndim != 2) {
                throw std::runtime_error("X-Data needs to have two dimensions");
            }
        };

        ~BufferAdaptor() {
            this->ref.dec_ref();
        }

        inline size_t kdtree_get_point_count() const {
            return source.shape[0];
        }

        inline size_t get_dim() const {
            return source.shape[1];
        }

        inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
            return *((T *) (((uint8_t *) (source.ptr)) + idx * source.strides[0] + dim * source.strides[1]));
        }

        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /*bb*/) const {
            return false;
        }

    private:
        pybind11::buffer_info source;
        pybind11::handle ref;

    };

    template<typename T>
    class NadarayaWatson {

        using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<T, BufferAdaptor<T>>,
                BufferAdaptor<T>>;

    public:
        NadarayaWatson(const pybind11::buffer &sx, const pybind11::buffer &sy, uint32_t n_threads = 1) :
                ref_y(sy.inc_ref()), source_y(sy.request()), adaptor(sx),
                tree(adaptor.get_dim(), adaptor, {10 /* max leaf */}), n_threads(n_threads) {
            if (source_y.format != pybind11::format_descriptor<T>::format()) {
                throw std::runtime_error("Expected y-data type " + pybind11::format_descriptor<T>::format() +
                                         "but got " + source_y.format);
            }

            if (source_y.ndim != 1) {
                throw std::runtime_error("Y-Data needs to have one dimension");
            }

            if (adaptor.kdtree_get_point_count() != source_y.shape[0]) {
                throw std::runtime_error("X- and Y-Data need to have the same number of entries");
            }
        };

        ~NadarayaWatson() {
            this->ref_y.dec_ref();
        };

        pybind11::array_t<T> predict(const pybind11::buffer &query_data, T bandwidth, uint32_t n_max = 200,
                                     T radius_scale = 3.) {
            pybind11::buffer_info info = query_data.request();
            if (info.format != pybind11::format_descriptor<T>::format()) {
                throw std::runtime_error("Expected file format " + pybind11::format_descriptor<T>::format() +
                                         " but got " + info.format);
            }

            if (info.shape[info.ndim - 1] != adaptor.get_dim()) {
                throw std::runtime_error("Query data needs to have the same number of dimensions as source data");
            }

            if (info.strides[info.ndim - 1] != sizeof(T)) {
                throw std::runtime_error("KD-Tree requires contiguous query array!");
            }

            T *res;
            size_t n_elem;
            if (info.ndim == 1) {
                n_elem = 1;
                res = new T[1];
                res[0] = this->predict_single((T *) info.ptr, bandwidth, n_max, radius_scale);
            } else if (info.ndim == 2) {
                n_elem = info.shape[0];
                res = new T[n_elem];
#pragma omp parallel for default(none) shared(res, n_elem, info, bandwidth, n_max, radius_scale) num_threads(this->n_threads)
                for (size_t i = 0; i < n_elem; i++) {
                    res[i] = predict_single((T *) (((uint8_t *) (info.ptr)) + i * info.strides[0]), bandwidth,
                                            n_max, radius_scale);
                }
            } else {
                throw std::runtime_error("Only 1- or 2-D arrays are supported for now!");
            }

            pybind11::capsule free_when_done(res, [](void *f) {
                T *foo = reinterpret_cast<T *>(f);
                delete[] foo;
            });

            // Wrap the result with a numpy array
            return pybind11::array_t<T>(
                    {n_elem}, // shape
                    {sizeof(T)}, // C-style contiguous strides for double
                    res, // the data pointer
                    free_when_done); // numpy array references this parent
        }

    private:
        size_t radiusSearch(const T *query_point, const T radius, std::vector<uint32_t> &indices,
                            std::vector<T> &dists, uint32_t n_max) const {
            LimitedRadiusResultSet<T, uint32_t> resultSet(radius, indices, dists, n_max);
            nanoflann::SearchParams params;
            this->tree.findNeighbors(resultSet, query_point, params);
            return resultSet.size();
        }

        inline T get_value(size_t idx) {
            return *((T *) (((uint8_t *) this->source_y.ptr) + idx * this->source_y.strides[0]));
        }

        inline T logsumexp(T *data, size_t size, T scale) {
            T max = -scale * (*std::min_element(data, data + size));

            T sum_exp = 0;
            for (size_t i = 0; i < size; ++i) {
                T log_weight = -scale * data[i];
                sum_exp += exp(log_weight - max);
                // So that we do not need to compute it again later
                data[i] = log_weight;
            }

            return log(sum_exp) + max;
        }


        T predict_single(void *data, T bw, uint32_t n_max, T radius_scale) {
            std::vector<uint32_t> indices;
            indices.reserve(10);
            std::vector<T> distances;
            distances.reserve(10);
            // nanoflann works with squared distances!
            T effective_radius = (radius_scale * bw) * (radius_scale * bw);
            size_t n_matches = this->radiusSearch((T *) data, effective_radius, indices, distances, n_max);

            T result = 0.;
            T scale = 1 / (2 * bw * bw);
            if (n_matches == 0) {
                n_matches = this->tree.knnSearch((T *) data, 10, &(indices[0]), &(distances[0]));
            }

            // After the call to logsumexp the distances array contains the unnormalized log weights
            T lse = logsumexp(&(distances[0]), n_matches, scale);
            for (uint32_t i = 0; i < n_matches; i++) {
                T weight = exp(distances[i] - lse);
                result += weight * this->get_value(indices[i]);
            }
            return result;
        }

        pybind11::buffer_info source_y;
        pybind11::handle ref_y;
        BufferAdaptor<T> adaptor;
        kd_tree_t tree;
        uint32_t n_threads;

    };

}


#endif //NADARAYA_WATSON_NADARAYA_WATSON_H
